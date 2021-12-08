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

from gensim.models import Word2Vec
model = Word2Vec.load("checkpoints/word2vec/word2vec.model")
sim3 = model.most_similar(u'美女', topn=20)
for key in sim3:
    print(key[0], key[1])
