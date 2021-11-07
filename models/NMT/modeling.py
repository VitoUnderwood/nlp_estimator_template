# coding=utf-8
import copy
import json

import tensorflow as tf

from models.model_utils import get_rnn_cell


class NMTConfig(object):
    @classmethod
    def from_dict(cls, json_object: dict):
        config = NMTConfig()
        for key, val in json_object.items():
            config.__dict__[key] = val
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NMT(object):
    def __init__(self, config, input_ids, output_ids, batch_size, pad=0, unk=1, end=2, start=3):
        """
        描述模型图结构
        :param config: 存放模型结构相关的参数
        :param input_ids: [batch_size, max_seq_length]
        """
        with tf.variable_scope("embedding"):
            word_embedding = tf.get_variable("word_embedding", shape=[config.vocab_size, config.embed_size])
            # [batch_size, max_seq_length, embed_size]
            inp_embed = tf.nn.embedding_lookup(word_embedding, input_ids)

            start_tokens = tf.tile([start], [batch_size])
            # 开头添加go，构造teacher forcing的输入
            teacher_input_ids = tf.concat([tf.expand_dims(start_tokens, 1), output_ids], 1)
            teacher_input_embed = tf.nn.embedding_lookup(word_embedding, teacher_input_ids)

        with tf.variable_scope("encoder"):
            fw_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob, scope="encoder")
            bw_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob, scope="encoder")
            # outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。当time_major = False时
            # output_fw和output_bw的形状为[batch_size,max_len,hidden_num]。在此情况下，
            # 最终的outputs可以用tf.concat([output_fw, output_bw],-1)
            # 这里面的[output_fw, output_bw]可以直接用outputs进行代替。
            # output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
            # output_state_fw和output_state_bw的类型为LSTMStateTuple，由（c,h）组成，分别代表memory cell 和hidden state.

            encoder_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                   cell_bw=bw_cell,
                                                                                   inputs=inp_embed,
                                                                                   dtype=tf.float32)
            # [batch_size, max_seq_length, hidden_size*2]
            encoder_outputs = tf.concat(encoder_outputs, -1)
            # [batch_size, hidden_size*2]
            encoder_final_state = tf.concat(encoder_final_state, -1)

        with tf.variable_scope("decoder"):
            # helper decoder dynamic_decode 配合使用
            # tf.contrib.seq2seq.TrainingHelper(inputs, sequence_length)
            # here use teacher forcing, the output as decoder's input, [batch_size, seq_len]，[EOS] id is 2
            teacher_input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(teacher_input_ids, 2)), 1)
            train_helper = tf.contrib.seq2seq.TrainingHelper(teacher_input_embed, teacher_input_lengths)

            # tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer=None)

            decoder_hidden_size = 2 * config.hidden_size

            cell = tf.contrib.rnn.GRUCell(num_units=decoder_hidden_size)

            input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input_ids, 2)), 1)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=decoder_hidden_size,
                                                                    memory=encoder_outputs,
                                                                    memory_sequence_length=input_lengths)

            attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                            attention_layer_size=decoder_hidden_size)

            out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, config.vocab_size)

            basic_decoder = tf.contrib.seq2seq.BasicDecoder(
                out_cell,
                train_helper,
                initial_state=out_cell.zero_state(dtype=tf.float32,
                                                  batch_size=batch_size).clone(cell_state=encoder_final_state))

            output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=basic_decoder,
                                                             impute_finished=True,
                                                             output_max_length=20)

            tf.logging.info("=========================================================================================")
            tf.logging.info(f"input_ids shape {input_ids.shape}")
            tf.logging.info(f"output_ids shape {output_ids.shape}")
            tf.logging.info(f"output shape {output.rnn_output.shape}")

            logits = output.rnn_output

            cali_output_ids = output_ids[:, :tf.shape(logits)[1]]

            # weights is mask for loss
            weights = tf.to_float(tf.not_equal(cali_output_ids, 2))

            # 这里必需要让两个tensor的形状一样，但是在output_id的长度并不是当前batch的数据的最大长度，而是实现预定的max_seq_len
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits, cali_output_ids, weights=weights)

            # self.loss = tf.reduce_sum(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(
            #         labels=cali_output_ids,
            #         logits=logits
            #     )
            # ) / batch_size
