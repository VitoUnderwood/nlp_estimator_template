# coding=utf-8

import json
import pickle

import jieba
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

from models.data2text.modeling import Model, ModelConfig
from models.model_utils import get_groups_output_2
from utils.text_process import merge_once, merge_sent

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

# sess = tf.Session()
# model_dir = 'checkpoints/data2text/model.ckpt-14269.meta'
# saver = tf.train.import_meta_graph(model_dir)
# saver.restore(sess, tf.train.latest_checkpoint(model_dir))

def model_fn_builder(model_config, learning_rate, word_vectors=None):
    """
    包装model_fn，传入额外的参数
    :param word_vectors: 用于初始化embedding层
    :param learning_rate: 控制优化器
    :param model_config: 用户自定义的模型性参数，区分model fn的config，是run config函数，参数固定，所以
    需要额外的config，模型参数用于创建模型
    """

    def model_fn(features, labels, mode, params, config):
        tf.logging.info("*** Features ***")

        for name in sorted(features.keys()):
            tf.logging.info(f"name = {name}, shape = {features[name].shape}")

        cate_id = features["cate_id"]
        key_input_ids = features["key_input_ids"]
        val_input_ids = features["val_input_ids"]
        text_ids = features["text_ids"]
        outputs_ids = features["outputs_ids"]
        groups_ids = features["groups_ids"]

        if not mode == tf.estimator.ModeKeys.TRAIN:
            model_config.keep_prob = 1

        model = Model(config=model_config,
                      cate_id=cate_id,
                      key_input_ids=key_input_ids,
                      val_input_ids=val_input_ids,
                      text_ids=text_ids,
                      outputs_ids=outputs_ids,
                      groups_ids=groups_ids,
                      beam_width=5,
                      maximum_iterations=32,
                      mode=mode,
                      word_vectors=word_vectors)

        # for train and eval
        if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
            elbo_loss = model.elbo_loss
            stop_loss = model.stop_loss
            bow_loss = model.bow_loss
            sent_loss = model.sent_loss
            group_loss = model.group_loss
            kl_loss = model.kl_div
            # kl_div = model.kl_
            loss = model.train_loss
            tf.summary.scalar('elbo_loss', elbo_loss)
            tf.summary.scalar('stop_loss', stop_loss)
            tf.summary.scalar('sent_loss', sent_loss)
            tf.summary.scalar('group_loss', group_loss)
            tf.summary.scalar('bow_loss', bow_loss)
            tf.summary.scalar('kl_loss', kl_loss)
            tf.summary.scalar('loss', loss)
        else:
            loss = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                    global_step=tf.train.get_global_step())
        else:
            train_op = None

        # only for eval
        # def metric_fn(per_example_loss, label_ids, logits):
        #     predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        #     # weights 做为mask 1 0
        #     accuracy = tf.metrics.accuracy(
        #         labels=label_ids, predictions=predictions)
        #     loss = tf.metrics.mean(values=per_example_loss)
        #     return {
        #         "eval_accuracy": accuracy,
        #         "eval_loss": loss,
        #     }
        #
        # eval_metrics = metric_fn(per_example_loss, label_ids, logits)

        # only for predict
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = model.predictions
        else:
            predictions = None

        # train_summary = tf.summary.merge([tf.summary.scalar("learning rate", learning_rate)])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

    return model_fn


model_config = ModelConfig.from_json_file('checkpoints/data2text/config.json')

run_config = tf.estimator.RunConfig(
    model_dir='checkpoints/data2text',
    keep_checkpoint_max=1,
    log_step_count_steps=100,
)
params = {
    'batch_size': 8
}

model_fn = model_fn_builder(
    model_config=model_config,
    learning_rate=2e-5,
    word_vectors=id2vec)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    params=params,
    config=run_config)



# export_dir = 'checkpoints/data2text/1645584302'
# predict_fn = predictor.from_saved_model(export_dir)

def serving_input_fn():
    # INFO:tensorflow:*** Features ***
    # INFO:tensorflow:name = cate_id, shape = (32, ?)
    # INFO:tensorflow:name = groups_ids, shape = (32, ?, ?)
    # INFO:tensorflow:name = key_input_ids, shape = (32, ?)
    # INFO:tensorflow:name = outputs_ids, shape = (32, ?, ?)
    # INFO:tensorflow:name = text_ids, shape = (32, ?)
    # INFO:tensorflow:name = val_input_ids, shape = (32, ?)
    cate_id = tf.placeholder(dtype=tf.int32, shape=[None, None], name='cate_id')
    text_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='text_ids')
    groups_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='groups_ids')
    key_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='key_input_ids')
    val_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='val_input_ids')
    outputs_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='outputs_ids')
    features = {
        'cate_id': cate_id,
        'key_input_ids': key_input_ids,
        'val_input_ids': val_input_ids,
        'text_ids': text_ids,
        'groups_ids': groups_ids,
        'outputs_ids': outputs_ids
    }
    return tf.estimator.export.ServingInputReceiver(features, features)


# predict_fn = tf.contrib.predictor.from_estimator(estimator, serving_input_fn)

print('模型加载完成 ...')

@app.route('/', methods=['POST'])
def ping_pong():
    line = """{"feature": [["类型", "裤"], ["风格", "简约"], ["风格", "潮"], ["图案", "格子"], ["图案", "几何"], ["图案", "线条"], ["裤长", "七分裤"], ["裤型", "阔腿裤"]], "title": "", "largeSrc": "http://gw.alicdn.com/imgextra/i2/646511815/TB2KBRHb1ySBuNjy1zdXXXPxFXa_!!646511815-0-beehive-scenes.jpg_790x10000Q75.jpg", "refSrc": "https://market.m.taobao.com/apps/market/content/index.html?&contentId=200551261363", "desc": "这 款 阔腿裤 ， 整体 设计 简约 利落 ， 时尚 的 阔 腿 款式 带来 鲜明 的 几何 设计 美感 ， 褪去 传统 装束 的 厚重 与 臃肿 ， 更具 轻盈 美感 。 搭配 七分裤 长 修饰 出 挺拔 的 腿部 线条 ， 气质 的 格纹 图案 不 显 单调 ， 尽显 女性 优雅 气质 。 斜门襟 设计 潮流 出众 ， 让 你 时刻 保持 动人 的 女性 风采 。", "file": "fcc77fd7d27d564aed705d99b33e6a39.jpg", "专有属性": [["裤长", "七分裤"], ["裤型", "阔腿裤"], ["类型", "裤"]], "共有属性": [["风格", "简约"], ["风格", "潮"], ["图案", "格子"], ["图案", "几何"], ["图案", "线条"]], "segment": {"seg_0": {"segId": 0, "key_type": ["裤型", "图案", "风格"], "order": [["风格", "简约"], ["裤型", "阔腿裤"], ["图案", "几何"]], "seg": "这 款 阔腿裤 ， 整体 设计 简约 利落 ， 时尚 的 阔 腿 款式 带来 鲜明 的 几何 设计 美感 ， 褪去 传统 装束 的 厚重 与 臃肿 ， 更具 轻盈 美感 。"}, "seg_1": {"segId": 1, "key_type": ["图案", "裤长"], "order": [["裤长", "七分裤"], ["图案", "线条"], ["图案", "格子"]], "seg": "搭配 七分裤 长 修饰 出 挺拔 的 腿部 线条 ， 气质 的 格纹 图案 不 显 单调 ， 尽显 女性 优雅 气质 。"}, "seg_2": {"segId": 2, "key_type": ["风格"], "order": [["风格", "潮"]], "seg": "斜门襟 设计 潮流 出众 ， 让 你 时刻 保持 动人 的 女性 风采 。"}}}
"""
    record = json.loads(line)
    # print(record)
    feats = record["feature"]
    keys = []
    vals = []
    for item in feats:
        keys.append(item[0])
        vals.append(item[1])
    cate = dict(record['feature'])['类型']
    text = list(jieba.cut("".join(record['desc'].split())))
    outputs = []
    groups = []
    for _, seg in record["segment"].items():
        order = [item[:2] for item in seg['order']]
        sent = list(jieba.cut("".join(seg['seg'].split())))
        if len(order) > 0 and len(sent) > 0:
            groups.append(order)
            outputs.append(sent)
    # cate = request.json.get('cate')
    keys = request.json.get('keys')
    vals = request.json.get('vals')
    # text = request.json.get('text')
    # outputs =
    # groups =
    print(type(keys))
    desc = ""
    # example = InputExample(keys, vals, desc)
    # print(cate)
    print(keys)
    print(vals)
    example = InputExample(cate, keys, vals, text, outputs, groups)

    feature = convert_single_example(example, key2id, val2id, word2id)
    # features = {
    #         'cate_id': cate_id,
    #         'key_input_ids': key_input_ids,
    #         'val_input_ids': val_input_ids,
    #         'text_ids': text_ids,
    #         'groups_ids': groups_ids,
    #         'outputs_ids': outputs_ids
    #     }

    # data = {"instances": [
    #     {
    #         'cate_id': feature.cate_id,
    #         'key_input_ids': feature.key_input_ids,
    #         'val_input_ids': feature.val_input_ids,
    #         'text_ids': feature.text_ids,
    #         'groups_ids': feature.groups_ids,
    #         'outputs_ids': feature.outputs_ids
    #     }
    # ]
    # }

    data = {
        'cate_id': np.array([feature.cate_id], dtype=np.int32),
        'key_input_ids': np.array([feature.key_input_ids], dtype=np.int32),
        'val_input_ids': np.array([feature.val_input_ids], dtype=np.int32),
        'text_ids': np.array([feature.text_ids], dtype=np.int32),
        'groups_ids': np.array([feature.groups_ids], dtype=np.int32),
        'outputs_ids': np.array([feature.outputs_ids], dtype=np.int32)
    }
    print(data)

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=data,
        shuffle=False)
    result = estimator.predict(input_fn=pred_input_fn)
    predictions = next(result)
    print("预测结果id", predictions)
    # predictions = predict_fn(data)
    # print(predictions['scores'])

    # param = json.dumps(data)
    # # res = requests.post('http://localhost:8502/v1/models/half_plus_two:predict', data=param)
    # print("post data ......")
    # # url = 'http://192.168.140.158:8501/v1/models/DT:predict'
    # url = 'http://localhost:8501/v1/models/DT:predict'
    # json_response = requests.post(url, data=param)
    # # text is a DICT
    # predictions = np.array(json.loads(json_response.text)["predictions"])

    # all_string_sent_cut, all_string_sent = get_out_put_from_tokens_beam_search(predictions, id2word)
    answer = get_groups_output_2(predictions, id2word)
    print(answer)
    answer = merge_once(answer, 10)
    answer = merge_sent(answer)
    print(answer)
    return jsonify({'answer': answer})  # （jsonify返回一个json格式的数据）


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, cate, keys, vals, text, outputs, groups):
        """Constructs a InputExample. """
        self.cate = cate
        self.keys = keys
        self.vals = vals
        self.text = text
        self.outputs = outputs
        self.groups = groups


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 cate_id,
                 key_input_ids,
                 val_input_ids,
                 text_ids,
                 outputs_ids,
                 groups_ids):
        self.cate_id = cate_id
        self.key_input_ids = key_input_ids
        self.val_input_ids = val_input_ids
        self.text_ids = text_ids
        self.outputs_ids = outputs_ids
        self.groups_ids = groups_ids


def convert_single_example(example: InputExample, key2id, val2id, word2id):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    # 不使用end token了 直接pad
    # 文本都是提前使用结巴进行分好的词，以及使用jieba结果训练的词向量
    cate = example.cate
    keys = example.keys
    vals = example.vals
    text = example.text
    outputs = example.outputs
    groups = example.groups

    key_val = list(zip(keys, vals))
    # 将分组后的kv和未进行分组的所有的kv进行映射, 使用list index即可
    groups_ids = []
    for order in groups:
        # groups_ids.append([key_val.index((k, v)) for k, v in order])
        groups_ids.append([0])
    # convert to id to compute
    cate_id = [val2id.get(cate, val2id["<UNK>"])]
    key_input_ids = [key2id.get(word, key2id["<UNK>"]) for word in keys]
    val_input_ids = [val2id.get(word, val2id["<UNK>"]) for word in vals]
    text_ids = [word2id.get(word, word2id["<UNK>"]) for word in text]
    outputs_ids = []
    for output in outputs:
        # teacher forcing has one new token start token, need a end token to alien
        # print(output)
        tmp = [word2id.get(word, word2id["<UNK>"]) for word in output]
        # print(tmp)
        tmp.append(word2id.get("<PAD>", word2id["<UNK>"]))
        outputs_ids.append(tmp)

    # 截断填充处理，主要针对的是output和group中第二维度不同的情况
    for item in [outputs_ids, groups_ids]:
        max_len = -1
        for lst in item:
            max_len = max(max_len, len(lst))
        for idx, lst in enumerate(item):
            if len(lst) < max_len:
                item[idx] = lst + [0] * (max_len - len(lst))
    # print("output", outputs_ids)
    # print('group', groups_ids)

    feature = InputFeatures(
        cate_id=cate_id,
        key_input_ids=key_input_ids,
        val_input_ids=val_input_ids,
        text_ids=text_ids,
        outputs_ids=outputs_ids,
        groups_ids=groups_ids)
    return feature


if __name__ == '__main__':
    print("启动服务器 ......")
    app.run(debug=True)
