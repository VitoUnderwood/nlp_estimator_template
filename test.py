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
