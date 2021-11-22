# coding=utf-8
import copy
import json

import tensorflow as tf

from models.model_utils import get_rnn_cell


class ModelConfig(object):
    @classmethod
    def from_dict(cls, json_object: dict):
        config = ModelConfig()
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


class QA_CVAE(object):
    def __init__(self, config, input_ids, output_ids, beam_width, maximum_iterations, mode, pad=0, unk=1,
                 end_token=2, start_token=3):
        """
        描述模型图结构
        :param config: 存放模型结构相关的参数
        :param input_ids: [batch_size, max_seq_length]
        """
        # 动态batch_size，方便train 和 infer 阶段使用，方便部署，直接写死会导致infer必需和train的时候batch_size一致
        batch_size = tf.shape(input_ids)[0]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope("embedding"):
            # 这里输入和输出有相同的重叠的部分，所以共享，否则只会增加softmax的计算量
            word_embedding = tf.get_variable("word_embedding", shape=[config.vocab_size, config.embed_size])
            # [batch_size, max_seq_length, embed_size]
            input_embed = tf.nn.embedding_lookup(word_embedding, input_ids)
            output_embed = tf.nn.embedding_lookup(word_embedding, output_ids)

            # 开头添加go，构造teacher forcing的输入
            start_tokens = tf.tile([start_token], [batch_size])
            teacher_input_ids = tf.concat([tf.expand_dims(start_tokens, 1), output_ids], 1)
            teacher_input_embed = tf.nn.embedding_lookup(word_embedding, teacher_input_ids)

        with tf.variable_scope("encoder"):
            with tf.variable_scope("input_encoder"):
                input_fw_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob,
                                             scope="input_encoder")
                input_bw_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob,
                                             scope="input_encoder")
                # outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。当time_major = False时
                # output_fw和output_bw的形状为[batch_size,max_len,hidden_num]。在此情况下，
                # 最终的outputs可以用tf.concat([output_fw, output_bw],-1)
                # 这里面的[output_fw, output_bw]可以直接用outputs进行代替。
                # output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
                # output_state_fw和output_state_bw的类型为LSTMStateTuple，由（c,h）组成，分别代表memory cell 和hidden state.

                input_encoder_outputs, input_encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=input_fw_cell,
                    cell_bw=input_bw_cell,
                    inputs=input_embed,
                    dtype=tf.float32)
                # [batch_size, max_seq_length, hidden_size*2]
                input_encoder_outputs = tf.concat(input_encoder_outputs, -1)
                # [batch_size, hidden_size*2]
                input_encoder_final_state = tf.concat(input_encoder_final_state, -1)
            with tf.variable_scope("output_encoder"):
                # use for CVAE
                output_fw_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob,
                                              scope="output_encoder")
                output_bw_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob,
                                              scope="output_encoder")
                output_encoder_outputs, output_encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=output_fw_cell,
                    cell_bw=output_bw_cell,
                    inputs=output_embed,
                    dtype=tf.float32)
                # [batch_size, max_seq_length, hidden_size*2]
                output_encoder_outputs = tf.concat(output_encoder_outputs, -1)
                # [batch_size, hidden_size*2]
                output_encoder_final_state = tf.concat(output_encoder_final_state, -1)

        with tf.variable_scope("CVAE"):
            # 先验网络, 用于infer阶段
            # [batch_size, hidden_size*2]
            prior_input = input_encoder_final_state
            prior_fc = tf.layers.dense(prior_input, config.hidden_size * 4, activation=tf.tanh)
            prior_fc_nd = tf.layers.dense(prior_fc, config.hidden_size * 4)
            # [batch_size, hidden_size*2]
            prior_mu, prior_log_sigma = tf.split(prior_fc_nd, 2, 1)
            # [batch_size, hidden_size*2]
            prior_z_state = self.sample_gaussian((batch_size, config.hidden_size * 2),
                                                 prior_mu,
                                                 prior_log_sigma)
            # 后验网络，在train的时候使用，用于构建decoder的输入，以及指导先验网络
            # [batch_size, hidden_size*4]
            post_input = tf.concat((input_encoder_final_state, output_encoder_final_state), 1)
            post_fc = tf.layers.dense(post_input, config.hidden_size * 4)
            post_mu, post_log_sigma = tf.split(post_fc, 2, 1)
            # [batch_size, hidden_size*2]
            post_z_state = self.sample_gaussian((batch_size, config.hidden_size * 2),
                                                post_mu,
                                                post_log_sigma)
            # 用作解码器的初始状态
            # [batch_size, hidden_size*4]
            if is_training:
                dec_input = tf.concat((input_encoder_final_state, post_z_state), 1)
            else:
                dec_input = tf.concat((input_encoder_final_state, prior_z_state), 1)

        with tf.variable_scope("decoder"):
            decoder_hidden_size = 4 * config.hidden_size
            cell = tf.contrib.rnn.GRUCell(num_units=decoder_hidden_size)
            projection = tf.layers.Dense(config.vocab_size)
            input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input_ids, 2)), 1)
            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                teacher_input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(teacher_input_ids, 2)), 1)

                train_helper = tf.contrib.seq2seq.TrainingHelper(teacher_input_embed, teacher_input_lengths)

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=decoder_hidden_size,
                                                                        memory=input_encoder_outputs,
                                                                        memory_sequence_length=input_lengths)

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                                attention_layer_size=decoder_hidden_size)

                decoder_init_state = attn_cell.zero_state(dtype=tf.float32,
                                                          batch_size=batch_size).clone(cell_state=dec_input)

                basic_decoder = tf.contrib.seq2seq.BasicDecoder(
                    attn_cell,
                    train_helper,
                    initial_state=decoder_init_state,
                    output_layer=projection)

                output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=basic_decoder,
                                                                 maximum_iterations=maximum_iterations)

                with tf.variable_scope("loss"):
                    # ELBO
                    kl_divergence = self.kl_divergence(prior_mu, prior_log_sigma, post_mu, post_log_sigma)

                    logits = output.rnn_output
                    cali_output_ids = output_ids[:, :tf.shape(logits)[1]]
                    # weights is mask for loss
                    weights = tf.to_float(tf.not_equal(cali_output_ids, 2))
                    # 这里必需要让两个tensor的形状一样，但是在output_id的长度并不是当前batch的数据的最大长度，而是实现预定的max_seq_len
                    rec_loss = tf.contrib.seq2seq.sequence_loss(logits, cali_output_ids, weights=weights)
                    # self.loss = tf.reduce_sum(
                    #     tf.nn.sparse_softmax_cross_entropy_with_logits(
                    #         labels=cali_output_ids,
                    #         logits=logits
                    #     )
                    # ) / batch_size
                    self.elbo_loss = rec_loss + kl_divergence

            if mode == tf.estimator.ModeKeys.PREDICT:
                encoder_outputs = tf.contrib.seq2seq.tile_batch(input_encoder_outputs, multiplier=beam_width)
                dec_input = tf.contrib.seq2seq.tile_batch(dec_input, multiplier=beam_width)
                input_lengths = tf.contrib.seq2seq.tile_batch(input_lengths, multiplier=beam_width)

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=decoder_hidden_size,
                                                                        memory=encoder_outputs,
                                                                        memory_sequence_length=input_lengths)

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                                attention_layer_size=decoder_hidden_size)

                decoder_initial_state = attn_cell.zero_state(dtype=tf.float32,
                                                             batch_size=batch_size * beam_width).clone(
                    cell_state=dec_input)

                beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(attn_cell,
                                                                    word_embedding,
                                                                    start_tokens,
                                                                    end_token,
                                                                    decoder_initial_state,
                                                                    beam_width=beam_width,
                                                                    output_layer=projection)

                output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=beam_decoder,
                                                                 maximum_iterations=maximum_iterations)

                self.predictions = output.predicted_ids

    @staticmethod
    def kl_divergence(prior_mu, prior_log_sigma, post_mu, post_log_sigma):
        divergence = 0.5 * tf.reduce_sum(tf.exp(post_log_sigma - prior_log_sigma)
                                         + tf.pow(post_mu - prior_mu, 2) / tf.exp(prior_log_sigma)
                                         - 1 - (post_log_sigma - prior_log_sigma), axis=1)
        return tf.reduce_mean(divergence)

    @staticmethod
    def sample_gaussian(shape, mu, log_sigma):
        x = tf.random_normal(shape, dtype=tf.float32)
        z = mu + tf.exp(log_sigma / 2) * x
        return z
