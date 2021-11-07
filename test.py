# -*- coding:utf-8 -*-
from models.bert import tokenization

tokenizer = tokenization.FullTokenizer(
      vocab_file="data/NMT/vocab.txt")
print(tokenizer.tokenize("hello world"))
print(tokenizer.convert_tokens_to_ids(['ausländer', 'world']))
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