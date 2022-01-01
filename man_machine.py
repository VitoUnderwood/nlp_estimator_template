# coding=utf-8

from flask import Flask, jsonify
from flask_cors import CORS

import json
import pickle
import numpy as np

import requests
from flask import request
from models.model_utils import get_out_put_from_tokens_beam_search
from utils.text_process import merge, merge_once, merge_sent

app = Flask(__name__)
CORS(app, resources={r"/": {"origins": "*"}})

print("load vocabulary ......")
id2word = pickle.load(open("checkpoints/data2text/vocab.pkl", "rb"))
id2vec = pickle.load(open("checkpoints/data2text/vector.pkl", "rb"))
id2key = pickle.load(open("checkpoints/data2text/key.pkl", "rb"))
id2val = pickle.load(open("checkpoints/data2text/val.pkl", "rb"))
print("load finish ......")

word2id = dict(zip(id2word, range(len(id2word))))
key2id = dict(zip(id2key, range(len(id2key))))
val2id = dict(zip(id2val, range(len(id2val))))


@app.route('/', methods=['POST'])
def ping_pong():
    keys = request.json.get('keys')
    vals = request.json.get('vals')
    print(type(keys))
    desc = ""
    example = InputExample(keys, vals, desc)

    feature = convert_single_example(example, key2id, val2id, word2id)

    data = {"instances": [
        {
            "key_input_ids": feature.key_input_ids,
            "val_input_ids": feature.val_input_ids,
            "output_ids": feature.output_ids
        }
    ]
    }

    param = json.dumps(data)
    # res = requests.post('http://localhost:8502/v1/models/half_plus_two:predict', data=param)
    print("post data ......")
    # url = 'http://192.168.140.158:8501/v1/models/DT:predict'
    url = 'http://localhost:8501/v1/models/DT:predict'
    json_response = requests.post(url, data=param)
    # text is a DICT
    predictions = np.array(json.loads(json_response.text)["predictions"])

    all_string_sent_cut, all_string_sent = get_out_put_from_tokens_beam_search(predictions, id2word)
    print(all_string_sent[0])
    answer = merge_once(all_string_sent[0], 10)
    answer = merge_sent(answer)
    print(answer)
    return jsonify({'answer': answer})  # （jsonify返回一个json格式的数据）


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, keys, vals, desc):
        """Constructs a InputExample. """
        self.keys = keys
        self.vals = vals
        self.desc = desc


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 key_input_ids,
                 val_input_ids,
                 output_ids):
        self.key_input_ids = key_input_ids
        self.val_input_ids = val_input_ids
        self.output_ids = output_ids


def convert_single_example(example, key2id, val2id, word2id, max_feat_num=5, max_seq_length=32):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    # 不使用end token了 直接pad
    keys = list(example.keys)
    vals = list(example.vals)
    desc = example.desc[4:]
    desc_tokens = desc.split()

    # 截断处理
    if len(desc_tokens) > max_seq_length:
        desc_tokens = desc_tokens[0:max_seq_length]

    if len(keys) > max_feat_num:
        keys = keys[0: max_feat_num]
        vals = vals[0: max_feat_num]

    # [PAD] 0 [S] 1 [E] 2
    # 由于做的是nlg，所以go用作第一个输入的token，在decode阶段需要使用，但是encode不需要
    output_tokens = []
    for token in desc_tokens:
        output_tokens.append(token)

    key_input_ids = [key2id.get(word, key2id["<UNK>"]) for word in keys]
    val_input_ids = [val2id.get(word, val2id["<UNK>"]) for word in vals]
    output_ids = [word2id.get(word, word2id["<UNK>"]) for word in output_tokens]

    while len(output_ids) < max_seq_length:
        output_ids.append(word2id["<PAD>"])

    while len(key_input_ids) < max_feat_num:
        key_input_ids.append(key2id["<PAD>"])

    while len(val_input_ids) < max_feat_num:
        val_input_ids.append(val2id["<PAD>"])

    assert len(output_ids) == max_seq_length
    assert len(key_input_ids) == max_feat_num
    assert len(val_input_ids) == max_feat_num

    feature = InputFeatures(
        key_input_ids=key_input_ids,
        val_input_ids=val_input_ids,
        output_ids=output_ids)
    return feature


if __name__ == '__main__':
    print("启动服务器 ......")
    app.run(debug=True)
