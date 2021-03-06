# coding=utf-8

import json
import os
import pickle

import jieba
import numpy as np
import tensorflow as tf

from models.data2text.modeling import Model, ModelConfig
from models.model_utils import get_groups_output

# try:
#     from tensorflow.python.util import module_wrapper as deprecation
# except ImportError:
#     from tensorflow.python.util import deprecation_wrapper as deprecation
# deprecation._PER_MODULE_WARNING_LIMIT = 0

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.get_logger()
logger.propagate = False

flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters, file path, super params defined in config.json
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "model_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("vector_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("key_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("val_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters
flags.DEFINE_integer(
    "beam_width", 5,
    "beam search parameter which control the search window size K")

flags.DEFINE_integer(
    "maximum_iterations", 50,
    "The maximum generation sequence length"
)

flags.DEFINE_integer(
    "max_feat_num", 10,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_early_stop_step", 10,
    "The maximum step of eval loss not decrease ")

flags.DEFINE_bool("do_train", False, "Whether to run train on the train set.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("save_model", False, "Whether to save the model to PB format for tensorflow serving")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")


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


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    batches could cause silent errors.
    """


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


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, set_type):
        """Reads a tab separated value file."""
        # with tf.gfile.Open(input_file, "r") as f:
        #     reader = csv.reader(f, delimiter="\t")
        #     lines = []
        #     for line in reader:
        #         lines.append(line)
        #     return lines
        tf.logging.info(f"{set_type}")
        with tf.gfile.Open(input_file, "r") as f:
            return f.readlines()


class Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json"), "train"))

    def get_dev_examples(self, data_dir):
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json"), "dev"))

    def get_test_examples(self, data_dir):
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json"), "test"))

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def create_examples(lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            record = json.loads(line)
            feats = record["feature"]
            keys = []
            vals = []
            for item in feats:
                keys.append(item[0])
                vals.append(item[1])
            cate = dict(record['feature'])['??????']
            text = list(jieba.cut("".join(record['desc'].split())))
            outputs = []
            groups = []
            for _, seg in record["segment"].items():
                order = [item[:2] for item in seg['order']]
                sent = list(jieba.cut("".join(seg['seg'].split())))
                if len(order) > 0 and len(sent) > 0:
                    groups.append(order)
                    outputs.append(sent)
            if len(keys) > 0 and len(text) > 0 and len(outputs) > 0 and len(cate) > 0:
                examples.append(InputExample(cate, keys, vals, text, outputs, groups))
            # print(examples[-1])
        return examples


def convert_single_example(example: InputExample, key2id, val2id, word2id):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    # ?????????end token??? ??????pad
    # ???????????????????????????????????????????????????????????????jieba????????????????????????
    cate = example.cate
    keys = example.keys
    vals = example.vals
    text = example.text
    outputs = example.outputs
    groups = example.groups

    key_val = list(zip(keys, vals))
    # ???????????????kv??????????????????????????????kv????????????, ??????list index??????
    groups_ids = []
    for order in groups:
        groups_ids.append([key_val.index((k, v)) for k, v in order])
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

    # ???????????????????????????????????????output???group??????????????????????????????
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


def file_based_convert_examples_to_features(examples, key2id, val2id, word2id, output_file):
    """Convert a set of `InputExample`s to a TFRecord file.
    ????????????????????????????????????????????????input features
    """

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info(f"Writing example {ex_index} of {len(examples)} for {output_file}")

        feature = convert_single_example(example, key2id, val2id, word2id)

        def create_int_feature(values):
            if isinstance(values, list):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
            else:
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))
            return f

        def create_matrix_feature(values):
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[np.array(values).astype(np.int32).tostring()]))

        features = {
            "cate_id": create_matrix_feature(feature.cate_id),
            "key_input_ids": create_matrix_feature(feature.key_input_ids),
            "val_input_ids": create_matrix_feature(feature.val_input_ids),
            "text_ids": create_matrix_feature(feature.text_ids),
            "outputs_ids": create_matrix_feature(feature.outputs_ids),
            "outputs_shape": create_matrix_feature([len(feature.outputs_ids), len(feature.outputs_ids[0])]),
            "groups_ids": create_matrix_feature(feature.groups_ids),
            "groups_shape": create_matrix_feature([len(feature.groups_ids), len(feature.groups_ids[0])])
        }
        # matrix????????????

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, is_training, drop_remainder):
    """
    train(input_fn)
    :param input_file: tf.record ???????????????
    :param is_training:
    :param drop_remainder: ????????????????????????????????????batch
    :return:
    """

    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        features = {
            "cate_id": tf.FixedLenFeature((), tf.string),
            "key_input_ids": tf.FixedLenFeature((), tf.string),
            "val_input_ids": tf.FixedLenFeature((), tf.string),
            "text_ids": tf.FixedLenFeature((), tf.string),
            "outputs_ids": tf.FixedLenFeature((), tf.string),
            "outputs_shape": tf.FixedLenFeature((), tf.string),
            "groups_ids": tf.FixedLenFeature((), tf.string),
            "groups_shape": tf.FixedLenFeature((), tf.string)
        }
        raw_example = tf.parse_single_example(record, features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        # for name in list(example.keys()):
        #     t = example[name]
        #     if t.dtype == tf.int64:
        #         t = tf.to_int32(t)
        #     example[name] = t
        example = {
            'cate_id': tf.decode_raw(raw_example['cate_id'], tf.int32),
            'key_input_ids': tf.decode_raw(raw_example['key_input_ids'], tf.int32),
            'val_input_ids': tf.decode_raw(raw_example['val_input_ids'], tf.int32),
            'text_ids': tf.decode_raw(raw_example['text_ids'], tf.int32),
            'outputs_ids': tf.reshape(tf.decode_raw(raw_example['outputs_ids'], tf.int32),
                                      tf.decode_raw(raw_example['outputs_shape'], tf.int32)),
            # 'outputs_shape': tf.decode_raw(raw_example['outputs_shape'], tf.int32),
            'groups_ids': tf.reshape(tf.decode_raw(raw_example['groups_ids'], tf.int32),
                                     tf.decode_raw(raw_example['groups_shape'], tf.int32)),
            # 'groups_shape': tf.decode_raw(raw_example['groups_shape'], tf.int32)
        }
        # return raw_example['cate_id'], raw_example['key_input_ids'], raw_example['val_input_ids'], raw_example[
        #     'text_ids'], tf.decode_raw(raw_example['outputs_ids'], tf.int32), tf.decode_raw(raw_example['groups_ids'],
        #                                                                                     tf.int32)
        return example

    def input_fn(params):
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        d = d.map(_decode_record)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=10000)

        # ????????????0??????????????? ???????????????padding_value??????
        # ?????????matrix????????????????????????????????????[[1,2,3],[1,2],[1]]??????????????????????????????????????????????????????????????????pad
        padded_shapes = {
            'cate_id': tf.TensorShape([None]),  # [] Scalar elements, no padding.
            'key_input_ids': tf.TensorShape([None]),  # [None] Vector elements, padded to longest.
            'val_input_ids': tf.TensorShape([None]),  # [None, None]Matrix elements, padded to longest
            'text_ids': tf.TensorShape([None]),
            'outputs_ids': tf.TensorShape([None, None]),
            'groups_ids': tf.TensorShape([None, None])
        }

        # padded_shapes = ([None], [None], [None], [None], [None, None], [None, None])
        d = d.padded_batch(batch_size=params["batch_size"],
                           padded_shapes=padded_shapes,
                           drop_remainder=drop_remainder)
        return d

    return input_fn


def model_fn_builder(model_config, learning_rate, word_vectors=None):
    """
    ??????model_fn????????????????????????
    :param word_vectors: ???????????????embedding???
    :param learning_rate: ???????????????
    :param model_config: ??????????????????????????????????????????model fn???config??????run config??????????????????????????????
    ???????????????config?????????????????????????????????
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
                      beam_width=FLAGS.beam_width,
                      maximum_iterations=FLAGS.maximum_iterations,
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
            kl_loss = model.anneal_kl_divergence
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
        #     # weights ??????mask 1 0
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

    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()


def main(_):
    # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    #     raise ValueError(
    #         "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    model_config = ModelConfig.from_json_file(FLAGS.model_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = Processor()

    tf.logging.info("load vocabulary ......")
    id2word = pickle.load(open(FLAGS.vocab_file, "rb"))
    id2vec = pickle.load(open(FLAGS.vector_file, "rb"))
    id2key = pickle.load(open(FLAGS.key_file, "rb"))
    id2val = pickle.load(open(FLAGS.val_file, "rb"))
    tf.logging.info("load finish ......")

    word2id = dict(zip(id2word, range(len(id2word))))
    key2id = dict(zip(id2key, range(len(id2key))))
    val2id = dict(zip(id2val, range(len(id2val))))

    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    model_fn = model_fn_builder(
        model_config=model_config,
        learning_rate=FLAGS.learning_rate,
        word_vectors=id2vec)

    # ?????????Estimator

    if FLAGS.do_train or FLAGS.do_eval:
        batch_size = FLAGS.train_batch_size
    else:
        batch_size = FLAGS.predict_batch_size

    params = {
        'batch_size': batch_size
    }

    # ??????????????????
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # ????????????save??????????????????
    # tf_random_seed = 2021,
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max=1,
        log_step_count_steps=100,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config)

    if FLAGS.do_train and FLAGS.do_eval:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")

        if not tf.gfile.Exists(train_file):
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            file_based_convert_examples_to_features(train_examples, key2id, val2id, word2id, train_file)
            tf.logging.info("  Num examples = %d", len(train_examples))

        if not tf.gfile.Exists(eval_file):
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
            file_based_convert_examples_to_features(
                eval_examples, key2id, val2id, word2id, eval_file)
            tf.logging.info("  Num examples = %d", len(eval_examples))

        train_input_fn = file_based_input_fn_builder(input_file=train_file, is_training=True, drop_remainder=True)

        eval_input_fn = file_based_input_fn_builder(input_file=eval_file, is_training=False, drop_remainder=True)

        # early stop hook ???????????????????????????????????????metric????????????
        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator,
                                                                                 metric_name="loss",
                                                                                 max_steps_without_decrease=FLAGS.max_early_stop_step)

        # ???train???????????????training.latest_checkpoint(checkpoint_dir))???saver.restore()??????????????????????????????????????????????????????????????????????????????????????????
        # ?????????????????????????????????????????????model dir???????????????
        # ?????????train spec???????????????max_steps=None ????????????????????????
        tf.estimator.train_and_evaluate(estimator,
                                        train_spec=tf.estimator.TrainSpec(train_input_fn, hooks=[early_stopping_hook]),
                                        eval_spec=tf.estimator.EvalSpec(eval_input_fn)
                                        )
        # train_and_evaluate ???????????????????????????summary???checkpoints???????????????eval?????????????????????

        # estimator.train(input_fn=train_input_fn, hooks=[hook], max_steps=num_train_steps)

        # result = estimator.evaluate(input_fn=eval_input_fn)
        #
        # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        # with tf.gfile.GFile(output_eval_file, "w") as writer:
        #     tf.logging.info("***** Eval results *****")
        #     for key in sorted(result.keys()):
        #         tf.logging.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))

        # save model for tensorflow service
        if FLAGS.save_model:
            estimator.export_saved_model(FLAGS.output_dir, serving_input_fn)

    if FLAGS.do_predict:
        if FLAGS.save_model:
            estimator.export_saved_model(FLAGS.output_dir, serving_input_fn)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        if not tf.gfile.Exists(predict_file):
            file_based_convert_examples_to_features(predict_examples, key2id, val2id, word2id, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(input_file=predict_file, is_training=False, drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        groups_outputs = get_groups_output(result, id2word)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")

        # print(num_actual_predict_examples)

        # print("==========================================================================================================================================================="+len(result))

        # assert num_actual_predict_examples == len(all_string_sent)

        with tf.gfile.GFile("checkpoints/data2text/demo.tsv", "w") as writer:
            for example in predict_examples:
                for key, val in zip(example.keys, example.vals):
                    writer.write(key + ":" + val + "\t")
                writer.write("\n")

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            for group in groups_outputs:
                for sent in group:
                    for word in sent:
                        writer.write(word)
                writer.write("\n")



if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("vector_file")
    flags.mark_flag_as_required("key_file")
    flags.mark_flag_as_required("val_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
