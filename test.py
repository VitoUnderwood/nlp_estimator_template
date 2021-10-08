# -*- coding:utf-8 -*-
from models.bert import tokenization

tokenizer = tokenization.FullTokenizer(
      vocab_file="checkpoints/chinese_L-12_H-768_A-12/vocab.txt")
print(tokenizer.wordpiece_tokenizer.tokenize("傻逼 学校"))
# examples = extract_features.read_examples("data/train.txt")
# print(examples)
# features = extract_features.convert_examples_to_features(
#       examples=examples, seq_length=128, tokenizer=tokenizer)
# print(features[0].tokens)
#
# import tensorflow as tf
#
# es = tf.estimator.LinearClassifier()
# es.train()
# es.evaluate()
# es.predict()
# es.export_savedmodel

with open("data/train.txt", 'r') as f:
      line = f.readline()
line.replace(r'\\', "\\")