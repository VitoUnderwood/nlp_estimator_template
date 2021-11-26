# # -*- coding:utf-8 -*-
# from models.bert import tokenization
#
# tokenizer = tokenization.FullTokenizer(
#       vocab_file="checkpoints/chinese_L-12_H-768_A-12/vocab.txt")
# print(tokenizer.tokenize("爱因斯坦26岁时就提出一个开创物理学新纪元的理论，这就是：	相对论啊——相对论是关于时空和引力的基本理论，主要由爱因斯坦(AlbertEinstein)创立，"))
# # examples = extract_features.read_examples("data/train.txt")
# # print(examples)
# # features = extract_features.convert_examples_to_features(
# #       examples=examples, seq_length=128, tokenizer=tokenizer)
# # print(features[0].tokens)
# #
# # import tensorflow as tf
# #
# # es = tf.estimator.LinearClassifier()
# # es.train()
# # es.evaluate()
# # es.predict()
# # es.export_savedmodel
import json

import requests

tmp = [1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545]

input_ids = []

for _ in range(16):
    input_ids.append(tmp)

output_ids = input_ids

data = {"instances": [
    {
        "input_ids": tmp,
        "output_ids": tmp
    }
]
}
# data = {"instances": [5, 7, 9]
# }

param = json.dumps(data)
# res = requests.post('http://localhost:8502/v1/models/half_plus_two:predict', data=param)
res = requests.post('http://192.168.140.158:8501/v1/models/QA:predict', data=param)
print(res.text)

# import tensorflow as tf
#
# tmp = [1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545,
#        1, 442, 5562, 2, 545]
#
# input_ids = []
#
# for _ in range(4):
#     input_ids.append(tmp)
#
# output_ids = input_ids
#
# MODEL_DIR = "checkpoints/QA_CVAE/1637496879"
#
# predict_fn = tf.contrib.predictor.from_saved_model(MODEL_DIR)
#
# # feature.*模型的输入
# prediction = predict_fn({
#     "input_ids": input_ids,
#     "output_ids": output_ids,
# })
# print(prediction)
# eval_logits = prediction["logits"]

# from gensim.models import Word2Vec
# model = Word2Vec.load("checkpoints/word2vec/word2vec.model")
# model.vocabulary
# # numpy vector of a word
# vector = model.wv["中国"]
# print(vector)
