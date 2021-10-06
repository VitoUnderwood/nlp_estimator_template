import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # tokenizer

        # load data here
        self.train_dataset = self.load_data(self.config.train_file, is_training=True)
        self.dev_dataset = self.load_data(self.config.dev_file, is_training=False)
        self.test_dataset = self.load_data(self.config.test_file, is_training=False)


    def load_data(self, filename, is_training):
        dataset = tf.data.TextLineDataset(filename)
        dataset = dataset.map(map_func=self.process)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        if is_training:
            # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
            dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size)

        # single data shape
        # tf.TensorShape([])     表示长度为单个数字
        # tf.TensorShape([None]) 表示长度未知的向量
        padded_shapes = (
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([])
        )
        # get a batch [batch_size, data_shape] 整合数据
        dataset.padded_batch(
            self.config.train_batch_size if is_training else self.config.test_batch_size,
            padded_shapes=padded_shapes
        )
        return dataset

    def map_fn(self, line):
        # 不需要使用py_func，数据预处理不属于图的一部分
        return tf.py_func(func=self.process, inp=[line], Tout=[tf.int32] * 14)

    def process(self, line):
        # 数据同一使用json字典格式存储，需要预先处理好
        line = json.loads(line.decode("utf-8"))
        # Convert word to ID
        return (
            np.array(key_input, dtype=np.int32),
            np.array(val_input, dtype=np.int32),
            np.array(input_lens, dtype=np.int32),
            np.array(target_input, dtype=np.int32),
            np.array(target_output, dtype=np.int32),
            np.array(output_lens, dtype=np.int32),
            np.array(group, dtype=np.int32),
            np.array(group_lens, dtype=np.int32),
            np.array(group_cnt, dtype=np.int32),
            np.array(target_type, dtype=np.int32),
            np.array(target_type_lens, dtype=np.int32),
            np.array(text, dtype=np.int32),
            np.array(slens, dtype=np.int32),
            np.array(category, dtype=np.int32),
        )

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.target[idx]


DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)

parent_dir = os.path.dirname(text_dir)


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)
