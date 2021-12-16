# -*- coding:utf-8 -*-
# from models.bert import tokenization
#
# tokenizer = tokenization.FullTokenizer(
#       vocab_file="checkpoints/word2vec/vocab.txt")
# print(tokenizer.tokenize("爱因斯坦26岁时就提出一个开创物理学新纪元的理论，这就是：	相对论啊——相对论是关于时空和引力的基本理论，主要由爱因斯坦(AlbertEinstein)创立，"))
# examples = extract_features.read_examples("data/train.txt")
# print(examples)
# features = extract_features.convert_examples_to_features(
#       examples=examples, seq_length=128, tokenizer=tokenizer)
# print(features[0].tokens)
#



# import json
#
# import requests
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
# for _ in range(16):
#     input_ids.append(tmp)
#
# output_ids = input_ids
#
# data = {"instances": [
#     {
#         "input_ids": tmp,
#         "output_ids": tmp
#     }
# ]
# }
# # data = {"instances": [5, 7, 9]
# # }
#
# param = json.dumps(data)
# # res = requests.post('http://localhost:8502/v1/models/half_plus_two:predict', data=param)
# res = requests.post('http://192.168.140.158:8501/v1/models/QA:predict', data=param)
# print(res.text)

import tensorflow as tf
A = tf.data.Dataset.range(1, 6).map(lambda x: tf.fill([x], x))
iterator = A.make_one_shot_iterator()
one_element = iterator.get_next()
B = A.padded_batch(2, padded_shapes=[-1])
iterator_B = B.make_one_shot_iterator()
one_element_B = iterator_B.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))
    for i in range(3):
        print(sess.run(one_element_B))
