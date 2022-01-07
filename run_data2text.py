# coding=utf-8

import collections
import json
import os
import pickle

import jieba
from tqdm import tqdm
import tensorflow as tf

from models.data2text.modeling import Model, ModelConfig
from models.model_utils import get_out_put_from_tokens_beam_search, get_out_put_from_tokens

try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.get_logger()
logger.propagate = False

flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters, file path
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
    "input_max_len", 20,
    "The maximum input sequence length")

flags.DEFINE_integer(
    "output_max_len", 20,
    "The maximum output sequence length")

flags.DEFINE_integer(
    "maximum_iterations", 128,
    "The maximum generation sequence length"
)
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

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

    def __init__(self, keys, vals, desc):
        """Constructs a InputExample. """
        self.keys = keys
        self.vals = vals
        self.desc = desc


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
                 key_input_ids,
                 val_input_ids,
                 output_ids):
        self.key_input_ids = key_input_ids
        self.val_input_ids = val_input_ids
        self.output_ids = output_ids


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
            for _, seg in record["segment"].items():
                keys = [record["feature"][0][0]]
                vals = [record["feature"][0][1]]
                features = seg["order"]
                desc = seg["seg"]
                for item in features:
                    keys.append(item[0])
                    vals.append(item[1])
                if len(keys) > 1 and len(vals) > 1 and len(desc) > 0:
                    examples.append(InputExample(keys, vals, desc))
        return examples


def convert_single_example(example, max_feat_num, max_seq_length, key2id, val2id, word2id):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    # 不使用end token了 直接pad
    keys = example.keys
    # print(keys)
    vals = example.vals
    desc_tokens = list(jieba.cut("".join(example.desc.split())))

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


def file_based_convert_examples_to_features(examples, max_feat_num, max_seq_length, key2id, val2id, word2id,
                                            output_file):
    """Convert a set of `InputExample`s to a TFRecord file.
    将输入和输出融和到一起构造模型的input features
    """

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info(f"Writing example {ex_index} of {len(examples)} for {output_file}")

        feature = convert_single_example(example, max_feat_num, max_seq_length, key2id, val2id, word2id)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["key_input_ids"] = create_int_feature(feature.key_input_ids)
        features["val_input_ids"] = create_int_feature(feature.val_input_ids)
        features["output_ids"] = create_int_feature(feature.output_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, max_feat_num, max_seq_length, is_training,
                                drop_remainder):
    """
    train(input_fn)
    :type max_feat_num: feature数量
    :param max_seq_length:
    :param input_file: tf.record 的文件路径
    :param is_training:
    :param drop_remainder: 是否保留最后一个不完整的batch
    :return:
    """

    name_to_features = {
        "key_input_ids": tf.FixedLenFeature([max_feat_num], tf.int64),
        "val_input_ids": tf.FixedLenFeature([max_feat_num], tf.int64),
        "output_ids": tf.FixedLenFeature([max_seq_length], tf.int64)}

    def _decode_record(record, features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=10000)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=params["batch_size"],
                drop_remainder=drop_remainder))

        return d

    return input_fn


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

        key_input_ids = features["key_input_ids"]
        val_input_ids = features["val_input_ids"]
        output_ids = features["output_ids"]

        if not mode == tf.estimator.ModeKeys.TRAIN:
            model_config.keep_prob = 1

        model = Model(config=model_config,
                      key_input_ids=key_input_ids,
                      val_input_ids=val_input_ids,
                      output_ids=output_ids,
                      beam_width=FLAGS.beam_width,
                      maximum_iterations=FLAGS.maximum_iterations,
                      mode=mode,
                      word_vectors=word_vectors)

        # for train and eval
        if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
            loss = model.elbo_loss
            rec_loss = model.rec_loss
            kl_div = model.kl_div
            tf.summary.scalar('elbo_loss', loss)
            tf.summary.scalar('kl_div', kl_div)
            tf.summary.scalar('rec_loss', rec_loss)
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


def serving_input_fn():
    key_input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_feat_num], name='key_input_ids')
    val_input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_feat_num], name='val_input_ids')
    output_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='output_ids')
    features = {
        'key_input_ids': key_input_ids,
        'val_input_ids': val_input_ids,
        'output_ids': output_ids
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

    # 普通的Estimator

    if FLAGS.do_train:
        batch_size = FLAGS.train_batch_size
    else:
        batch_size = FLAGS.predict_batch_size

    params = {
        'batch_size': batch_size
    }

    # 单机多卡训练
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # 定义模型save的频率，路径
    # tf_random_seed = 2021,
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max=3,
        log_step_count_steps=100,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config)

    if FLAGS.do_train and FLAGS.do_eval:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")

        train_examples = processor.get_train_examples(FLAGS.data_dir)

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)

        if not tf.gfile.Exists(train_file):
            file_based_convert_examples_to_features(
                train_examples, FLAGS.max_feat_num, FLAGS.max_seq_length,
                key2id, val2id, word2id, train_file)
        if not tf.gfile.Exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, FLAGS.max_feat_num, FLAGS.max_seq_length,
                key2id, val2id, word2id, eval_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            max_feat_num=FLAGS.max_feat_num,
            max_seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("\tNum examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("\tBatch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            max_feat_num=FLAGS.max_feat_num,
            max_seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True)

        # early stop hook
        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator,
                                                                                 metric_name="loss",
                                                                                 max_steps_without_decrease=FLAGS.max_early_stop_step,
                                                                                 run_every_steps=1,
                                                                                 run_every_secs=None)

        # 在train函数中存在training.latest_checkpoint(checkpoint_dir))，saver.restore()所以说每一次都会自动加载与模型匹配的参数，如果模型发生变化，
        # 会报错，这时候需要重新定义一个model dir保存新模型
        # 可以在train spec中添加参数max_steps=None 控制最大训练步数
        tf.estimator.train_and_evaluate(estimator,
                                        train_spec=tf.estimator.TrainSpec(train_input_fn, hooks=[early_stopping_hook]),
                                        eval_spec=tf.estimator.EvalSpec(eval_input_fn)
                                        )
        # train_and_evaluate 只会保留训练过程的summary和checkpoints，想要保存eval结果需要自己写

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
        estimator.export_saved_model(FLAGS.output_dir, serving_input_fn)

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        if not tf.gfile.Exists(predict_file):
            file_based_convert_examples_to_features(predict_examples, FLAGS.max_feat_num, FLAGS.max_seq_length,
                                                    key2id, val2id, word2id, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            max_feat_num=FLAGS.max_feat_num,
            max_seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        all_string_sent_cut, all_string_sent = get_out_put_from_tokens_beam_search(result, id2word)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")

        print(num_actual_predict_examples)

        # print("==========================================================================================================================================================="+len(result))

        # assert num_actual_predict_examples == len(all_string_sent)

        with tf.gfile.GFile("checkpoints/data2text/demo.tsv", "w") as writer:
            for example in predict_examples:
                for key, val in zip(example.keys, example.vals):
                    writer.write(key + ":" + val + "\t")
                writer.write("\n")

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            for each_answer in all_string_sent:
                writer.write(each_answer + "\n")

        estimator.export_saved_model(FLAGS.output_dir, serving_input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("vector_file")
    flags.mark_flag_as_required("key_file")
    flags.mark_flag_as_required("val_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
