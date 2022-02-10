# coding=utf-8
import copy
import json

import numpy as np
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


class Model(object):
    def __init__(self, config, cate_id, key_input_ids, val_input_ids, text_ids, outputs_ids,
                 groups_ids, beam_width, maximum_iterations, mode,
                 word_vectors=None, pad=0, end_token=2, start_token=1):
        """
        描述模型图结构
        <PAD>', '<S>', '<E>'
        """
        # 动态batch_size，方便train 和 infer 阶段使用，方便部署，直接写死会导致infer必需和train的时候batch_size一致
        self.config = config
        batch_size = tf.shape(cate_id)[0]
        # 控制数据的切换，不影响图的结构
        is_training = (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)

        with tf.variable_scope("encoder"):
            with tf.variable_scope("embedding"):
                # 这里输入和输出有相同的重叠的部分，所以共享，否则只会增加softmax的计算量
                if word_vectors is None:
                    word_embedding = tf.get_variable("word_embedding", shape=[config.vocab_size, config.embed_size])
                else:
                    word_embedding = tf.get_variable("word_embedding", dtype=tf.float32,
                                                     initializer=tf.constant(word_vectors, dtype=tf.float32))

                key_embedding = tf.get_variable("key_embedding", shape=[config.key_size, config.embed_size])
                val_embedding = tf.get_variable("val_embedding", shape=[config.val_size, config.embed_size])

                # [batch_size, max_feat_num, embed_size]
                key_embed = tf.nn.embedding_lookup(key_embedding, key_input_ids)
                # [batch_size, max_feat_num, embed_size]
                val_embed = tf.nn.embedding_lookup(val_embedding, val_input_ids)
                input_embed = tf.concat([key_embed, val_embed], -1)
                # val_mean_embed = tf.reduce_mean(val_embed, 2)
                text_embed = tf.nn.embedding_lookup(word_embedding, text_ids)

                # 开头添加go，构造teacher forcing的输入
                # start_tokens = tf.tile([start_token], [batch_size])
                # teacher_input_ids = tf.concat([tf.expand_dims(start_tokens, 1), outputs_ids], 1)
                # teacher_input_embed = tf.nn.embedding_lookup(word_embedding, teacher_input_ids)
                # input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(key_input_ids, pad)), 1)
                # output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(text_ids, pad)), 1)
                start_tokens = tf.tile([[start_token]], [batch_size, tf.shape(outputs_ids)[1]])
                groups_teacher_input_ids = tf.concat([tf.expand_dims(start_tokens, 2), outputs_ids], 2)
                groups_teacher_input_embed = tf.nn.embedding_lookup(word_embedding, groups_teacher_input_ids)

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
                # 返回的如果是双向rnn, 第一个是前向, 第二个是后向 [2, batch_size, max_seq_len, hidden_size]
                # 如果是多层rnn 那么outputs不变, final state变成n_layer的list[2, list]
                input_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(key_input_ids, pad)), 1)
                input_encoder_outputs, input_encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=input_fw_cell,
                    cell_bw=input_bw_cell,
                    inputs=input_embed,
                    sequence_length=input_lens,
                    dtype=tf.float32)
                # output[2, batch_size, seq_len, hidden_size], final_state [2, num_layers, batch_size, hidden_size]
                # [batch_size, max_seq_length, hidden_size*2] , [batch_size, hidden_size*2]
                # axis = sum(axis(i))
                input_encoder_outputs = tf.concat(input_encoder_outputs, -1)
                if config.num_rnn_layers > 1:
                    input_encoder_final_state = tf.concat(
                        (input_encoder_final_state[0][-1], input_encoder_final_state[1][-1]), -1)
                else:
                    input_encoder_final_state = tf.concat(input_encoder_final_state, -1)

            with tf.variable_scope("text_encoder"):
                # use for CVAE
                text_fw_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob,
                                            scope="text_encoder")
                text_bw_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob,
                                            scope="text_encoder")
                text_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(text_ids, pad)), 1)
                text_encoder_outputs, text_encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=text_fw_cell,
                    cell_bw=text_bw_cell,
                    inputs=text_embed,
                    sequence_length=text_lens,
                    dtype=tf.float32)
                # [batch_size, max_seq_length, hidden_size*2]
                output_encoder_outputs = tf.concat(text_encoder_outputs, -1)
                # [batch_size, hidden_size*2]
                if config.num_rnn_layers > 1:
                    text_encoder_final_state = tf.concat(
                        (text_encoder_final_state[0][-1], text_encoder_final_state[1][-1]), -1)
                else:
                    text_encoder_final_state = tf.concat(text_encoder_final_state, -1)

            with tf.variable_scope("CVAE"):
                # 先验网络, 用于infer阶段
                # [batch_size, hidden_size*4]
                prior_fc = tf.layers.dense(input_encoder_final_state, config.hidden_size * 4, activation=tf.tanh)
                prior_fc_nd = tf.layers.dense(prior_fc, config.hidden_size * 4)
                # [batch_size, hidden_size*2]
                prior_mu, prior_log_sigma = tf.split(prior_fc_nd, 2, 1)
                # [batch_size, hidden_size*2]
                prior_z_state = self.sample_gaussian((batch_size, config.hidden_size * 2),
                                                     prior_mu,
                                                     prior_log_sigma)
                # 后验网络，在train的时候使用，用于构建decoder的输入，以及指导先验网络
                # [batch_size, hidden_size*4]
                post_input = tf.concat((input_encoder_final_state, text_encoder_final_state), 1)
                post_fc = tf.layers.dense(post_input, config.hidden_size * 4)
                post_mu, post_log_sigma = tf.split(post_fc, 2, 1)
                # [batch_size, hidden_size*2]
                post_z_state = self.sample_gaussian((batch_size, config.hidden_size * 2),
                                                    post_mu,
                                                    post_log_sigma)
                # 用作解码器的初始状态
                # [batch_size, hidden_size*4]
                # dec_input = tf.concat((input_encoder_final_state, post_z_state), 1) if is_training else tf.concat(
                #     (input_encoder_final_state, prior_z_state), 1)
                if is_training:
                    dec_input = tf.concat((input_encoder_final_state, post_z_state), 1)
                else:
                    dec_input = tf.concat((input_encoder_final_state, prior_z_state), 1)
                # 不使用CVAE结构
                # dec_input = input_encoder_final_state

            group_decoder = get_rnn_cell("gru", config.num_rnn_layers, 2 * config.hidden_size, config.keep_prob,
                                         scope="infer_grouping")
            group_fc_1 = tf.layers.Dense(config.hidden_size)
            group_fc_2 = tf.layers.Dense(1)
            # 用于判断是否要继续进行下一个分组, 1停止，0继续
            stop_fc = tf.layers.Dense(1)
            group_init_state_fc = tf.layers.Dense(config.hidden_size * 2)
            init_input = tf.get_variable(name="start_of_group",
                                         initializer=tf.truncated_normal_initializer(),
                                         shape=(1, 2 * config.hidden_size),
                                         dtype=tf.float32)

            with tf.variable_scope("group_encoder"):
                # 根据group id将对应的encode结果取出来,
                # [batch_size, groups_cnt, group_len]
                # group_shape = tf.shape(groups_ids)
                # # 扩充维度 [batch_size, 1, 1]
                # batch_idx = tf.expand_dims(tf.expand_dims(tf.range(batch_size), 1), 2)
                # # 第一维拓展1倍，第二维拓展groups_cnt倍，第三维拓展到group_len倍 [batch_size,groups_cnt, group_len]
                # batch_idx = tf.tile(batch_idx, [1, group_shape[1], group_shape[2]])
                # # [batch_size,groups_cnt, group_len, 1]
                # batch_idx = tf.expand_dims(batch_idx, 3)
                # # [batch_size, groups_cnt, group_len, 1]
                # feat_idx = tf.expand_dims(groups_ids, 3)
                # # [batch_size, groups_cnt, feat_len, 2]
                # group_feat_idx = tf.concat((batch_idx, feat_idx), 3)
                # # 第i个batch的第j个feat, 目的是取出分组后id对应的kv encode的结果, [batch_size, groups_cnt, feat_cnt, hidden_size]
                # group_embed = tf.gather_nd(input_encoder_outputs, group_feat_idx)
                # group_mask = tf.not_equal(groups_ids, pad)
                # expanded_group_mask = tf.expand_dims(group_mask, 3)
                # # [batch_size, groups_cnt, hidden_size] reduce sum rank 自动squeeze
                # group_encode_sum = tf.reduce_sum(group_embed * expanded_group_mask, 2)
                # # [batch_size, groups_cnt]
                # group_lens = tf.reduce_sum(tf.to_int32(group_mask), 1)
                # # safe_group_lens = group_lens + tf.cast(tf.equal(group_lens, 0), dtype=tf.int32)
                # # [batch_size, groups_cnt, hidden_size]
                # group_mean_encode = group_encode_sum / tf.to_float(tf.expand_dims(group_lens, 2))

                group_encoder = get_rnn_cell("gru",
                                             config.num_rnn_layers,
                                             config.hidden_size,
                                             config.keep_prob,
                                             "group_encoder")

                groups_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(groups_ids, pad)), 2)
                groups_cnt = tf.reduce_sum(tf.to_int32(tf.not_equal(groups_lens, pad)), 1)
                gidx, group_bow, group_mean_bow, group_embed = self.gather_group(input_encoder_outputs,
                                                                                 groups_ids,
                                                                                 groups_lens,
                                                                                 groups_cnt,
                                                                                 group_encoder)

                # infer_groups_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(infer_groups_ids, pad)), 2)
                # infer_groups_cnt = tf.reduce_sum(tf.to_int32(tf.not_equal(infer_groups_lens, pad)), 1)
                # infer_gidx, infer_group_bow, infer_group_mean_bow, infer_group_embed = self.gather_group(
                #     input_encoder_outputs,
                #     infer_groups_ids,
                #     infer_groups_lens,
                #     infer_groups_cnt,
                #     group_encoder)

        with tf.variable_scope("decoder"):
            # 解码的时候需要进行train和predict的区分，采取的策略不同
            # 共享结构声明定义
            decoder_hidden_size = dec_input.shape.as_list()[-1]
            cell = tf.contrib.rnn.GRUCell(num_units=decoder_hidden_size)
            projection = tf.layers.Dense(config.vocab_size)
            bow_fc_1 = tf.layers.Dense(config.hidden_size)
            bow_fc_2 = tf.layers.Dense(config.vocab_size)
            decoder_gru_cell = get_rnn_cell("gru", config.num_rnn_layers, config.hidden_size, config.keep_prob,
                                            scope="decoder")
            input_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(key_input_ids, pad)), 1)

            with tf.variable_scope("train_decoder"):
                # 分组之后，每一条数据都由多个分组组成，每个分组生成一段话，每句话之间同样有依赖
                def train_cond(i, group_input, group_state, group_loss, stop_loss, sent_loss, kl_loss,
                               bow_loss):
                    return i < tf.shape(groups_ids)[1]

                def train_body(i, group_input, group_state, group_loss, stop_logits, sent_loss, kl_loss,
                               bow_loss):
                    teacher_input_embed = groups_teacher_input_embed[:, i, :]
                    # batch_size, feat_len
                    teacher_input_ids = groups_teacher_input_ids[:, i, :]
                    teacher_input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(teacher_input_ids, pad)), 1)

                    tf.reduce_sum(tf.to_int32(tf.not_equal(key_input_ids, pad)), 1)
                    sent_output = outputs_ids[:, i, :]

                    with tf.name_scope("grouping"):
                        sent_feat_ids = gidx[:, i, :, :]
                        sent_feat_bow = group_bow[:, i, :, :]
                        sent_feat_lens = groups_lens[:, i]
                        safe_sent_feat_lens = sent_feat_lens + tf.cast(tf.equal(sent_feat_lens, 0), dtype=tf.int32)
                        group_mask = tf.sequence_mask(sent_feat_lens, tf.shape(sent_feat_bow)[1], dtype=tf.float32)
                        expanded_group_mask = tf.expand_dims(group_mask, 2)
                        loss_mask = tf.to_float(tf.not_equal(sent_feat_lens, 0))

                        group_output, group_state = group_decoder(group_input, group_state)
                        tile_gout = tf.tile(tf.expand_dims(group_output, 1), [1, tf.shape(input_encoder_outputs)[1], 1])
                        # 判断每一个feat是否属于当前i分组
                        group_fc_input = tf.concat((input_encoder_outputs, tile_gout), 2)
                        group_logit = tf.squeeze(group_fc_2(tf.tanh(group_fc_1(group_fc_input))), 2)
                        group_label = tf.one_hot(sent_feat_ids[:, :, 1], tf.shape(group_logit)[1], dtype=tf.float32)
                        group_label = tf.reduce_sum(group_label * expanded_group_mask, 1)
                        group_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=group_label,
                                                                                      logits=group_logit)
                        input_mask = tf.sequence_mask(input_lens, tf.shape(group_logit)[1], dtype=tf.float32)
                        group_cross_entropy = loss_mask * tf.reduce_sum(group_cross_entropy * input_mask, 1)
                        group_loss += tf.reduce_sum(group_cross_entropy)
                        # 判断是否停止
                        stop_logits = stop_logits.write(i, tf.squeeze(stop_fc(group_output), axis=1))

                    with tf.name_scope("train_sent_decoder"):
                        sent_dec_state = group_mean_bow[:, i, :]
                        # sent_ids = group_feat_idx[:, i, :, :]
                        # sent_input_embed = group_embed[:, i, :, :]
                        with tf.variable_scope("LuongAttention"):
                            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=decoder_hidden_size,
                                                                                    memory=sent_feat_bow,
                                                                                    memory_sequence_length=sent_feat_lens)
                        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_gru_cell, attention_mechanism,
                                                                           attention_layer_size=decoder_hidden_size)
                        train_decoder_state = decoder_cell.zero_state(batch_size, dtype=tf.float32).clone(
                            cell_state=sent_dec_state)
                        # 确定输入
                        # >>>>>>>>>>>>>>>>>>>>此处需要考虑将gru的输出编码进来，加强依赖<<<<<<<<<<<<<<<<<<<<<
                        train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=teacher_input_embed,
                                                                         sequence_length=teacher_input_lengths)
                        basic_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                        helper=train_helper,
                                                                        initial_state=train_decoder_state,
                                                                        output_layer=projection)
                        # 类似rnn的dynamic
                        fout, fstate, flens = tf.contrib.seq2seq.dynamic_decode(basic_decoder, impute_finished=True)

                        # loss计算
                        # batch_size, seq_len
                        sent_logits = fout.rnn_output
                        cali_sent_output = sent_output[:, :tf.shape(sent_logits)[1]]
                        sent_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cali_sent_output,
                                                                                            logits=sent_logits)
                        sent_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(sent_output, pad)), 1)
                        sent_mask = tf.sequence_mask(sent_lens, tf.shape(sent_output)[1], dtype=tf.float32)
                        cali_sent_mask = sent_mask[:, :tf.shape(sent_logits)[1]]
                        sent_cross_entropy = loss_mask * tf.reduce_sum(sent_cross_entropy * cali_sent_mask, axis=1)
                        sent_loss += tf.reduce_sum(sent_cross_entropy)  # / effective_cnt
                    with tf.name_scope("bow_loss"):
                        bow_logits = bow_fc_2(tf.tanh(bow_fc_1(sent_dec_state)))
                        bow_logits = tf.tile(tf.expand_dims(bow_logits, 1), [1, tf.shape(sent_output)[1], 1])
                        bow_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sent_output,
                                                                                           logits=bow_logits)
                        bow_cross_entropy = loss_mask * tf.reduce_sum(bow_cross_entropy * sent_mask, axis=1)
                        bow_loss += tf.reduce_sum(bow_cross_entropy)  # / effective_cnt

                    return i + 1, group_input, group_state, group_loss, stop_logits, sent_loss, kl_loss, bow_loss

                group_input = tf.tile(init_input, [batch_size, 1])
                group_state = group_init_state_fc(dec_input)
                group_state = tuple(group_state for _ in range(config.num_rnn_layers))

                stop_logits = tf.TensorArray(dtype=tf.float32, element_shape=(None,), size=tf.shape(groups_ids)[1])

                _, group_input, group_state, group_loss, stop_logits, sent_loss, kl_loss, bow_loss, predicted_ids = tf.while_loop(
                    train_cond,
                    train_body,
                    loop_vars=(0, group_input, group_state, 0, stop_logits, 0, 0, 0))

                with tf.name_scope("loss_computation"):
                    stop_logits = tf.transpose(stop_logits.stack(), [1, 0])
                    groups_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(groups_ids, pad)), 2)
                    stop_label = tf.one_hot(groups_lens - 1, tf.shape(stop_logits)[1])
                    stop_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=stop_logits, labels=stop_label)
                    stop_mask = tf.sequence_mask(groups_lens, tf.shape(stop_logits)[1], dtype=tf.float32)
                    self.stop_loss = tf.reduce_mean(tf.reduce_sum(stop_cross_entropy * stop_mask, 1))

                    self.sent_loss = sent_loss / tf.to_float(batch_size)
                    self.group_loss = group_loss / tf.to_float(batch_size)
                    # self.sent_KL_divergence = KL_loss
                    # sent_KL_weight = tf.minimum(1.0, tf.to_float(self.global_step) / self.config.PHVM_sent_full_KL_step)
                    # self.type_loss = type_loss
                    self.bow_loss = bow_loss / tf.to_float(batch_size)
                    # anneal_sent_KL = sent_KL_weight * self.sent_KL_divergence
                    # anneal_plan_KL = plan_KL_weight * self.plan_KL_divergence
                    self.elbo_loss = sent_loss + group_loss  # + self.sent_KL_divergence + self.plan_KL_divergence
                    self.elbo_loss = self.elbo_loss / tf.to_float(batch_size)
                    # self.anneal_elbo_loss = self.sent_rec_loss + self.group_rec_loss + self.type_loss + \
                    #                         anneal_sent_KL + anneal_plan_KL
                    # self.anneal_elbo_loss /= tf.to_float(self.batch_size)
                    # self.train_loss = self.anneal_elbo_loss + self.stop_loss + self.bow_loss
                    self.train_loss = self.elbo_loss + self.stop_loss + self.bow_loss
                # output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=basic_decoder, impute_finished=True)

                # with tf.variable_scope("loss"):
                #     # ELBO
                #     kl_weight = tf.minimum(1.0, tf.to_float(tf.train.get_global_step()) / config.kl_annealing_step)
                #     self.kl_div = self.kl_divergence(prior_mu, prior_log_sigma, post_mu, post_log_sigma)
                #     self.anneal_kl_divergence = kl_weight * self.kl_div
                #
                #     # 正常来说teacher input 的长度多了一个<start>
                #     logits = output.rnn_output[:, :-1]
                #     # 调整logit 和 label 的 shape 一样后才能进行计算
                #     cali_output_ids = output_ids[:, :tf.shape(logits)[1]]
                #     # weights is mask for loss
                #     weights = tf.to_float(tf.not_equal(cali_output_ids, pad))
                #     # 这里必需要让两个tensor的形状一样，但是在logit相对output_id的长度多了一个<start>token， 所以，不妨去掉logit最后一个
                #     self.rec_loss = tf.contrib.seq2seq.sequence_loss(logits, cali_output_ids, weights=weights)
                #     # self.loss = tf.reduce_sum(
                # #     tf.nn.sparse_softmax_cross_entropy_with_logits(
                # #         labels=cali_output_ids,
                # #         logits=logits
                # #     )
                # # ) / batch_size
                # # self.elbo_loss = self.rec_loss + self.kl_div
                #     self.elbo_loss = self.rec_loss + self.anneal_kl_divergence

            with tf.variable_scope("infer_grouping"):
                def group_cond(i, group_input, group_state, infer_groups_ids, stop):
                    # 第 i=0 组，直接ture，进行生成第0组，否则，判断stop == 0, 存在需要继续生成的组，没有到达最大分组数量，生成i组
                    # stop 中 等于0的元素代表继续生成， 非0代表停止的时候的长度
                    return tf.cond(tf.equal(i, 0),
                                   lambda: True,
                                   lambda: tf.cond(tf.equal(tf.reduce_min(stop), 0),
                                                   lambda: tf.cond(
                                                       tf.less(i, config.max_group_cnt),
                                                       lambda: True,
                                                       lambda: False),
                                                   lambda: False))

                def group_body(i, group_input, group_state, infer_groups_ids, stop):
                    # [batch_size, feat_cnt, 2], 第i组特征对应的id
                    # seg_feat_idx = feat_idx[:, i, :, :]
                    # [batch_size, hidden_size]
                    # group_embed = group_mean_encode[:, i, :]
                    # [batch_size, 2*hidden_size], [batch_size, 2*hidden_size]
                    group_output, group_state = group_decoder(group_input, group_state)
                    # [batch_size]
                    stop_next = tf.greater(tf.sigmoid(tf.squeeze(stop_fc(group_output), axis=1)),
                                           config.stop_threshold)
                    # 0/1 * 0/1 * i+1 继续生成的话 stop 对应元素为0，否则的话记录下停止的时候的长度i+1
                    stop += tf.cast(tf.equal(stop, 0), dtype=tf.int32) * tf.cast(stop_next, dtype=tf.int32) * (
                            i + 1)
                    # 拓展复制到原始input的维度，对应到每一个feature, [batch_size, all_feat_len, hidden_size]，然后分配到每一个feat上
                    tile_gout = tf.tile(tf.expand_dims(group_output, 1), [1, tf.shape(input_encoder_outputs)[1], 1])
                    group_fc_input = tf.concat((input_encoder_outputs, tile_gout), 2)
                    # [batch_size, feat_len# loss注意输入不需要经过sigmoid]
                    group_logits = tf.squeeze(group_fc_2(tf.tanh(group_fc_1(group_fc_input))), 2)
                    input_mask = tf.not_equal(key_input_ids, pad)
                    group_probs = tf.sigmoid(group_logits) * input_mask

                    # 根据每一个输入对象计算得到的prob为当前分组选择id
                    gids, glens = tf.py_func(self.select, [group_probs, tf.shape(input_encoder_outputs)[1]],
                                             [tf.int32, tf.int32])
                    # [batch_size, group_feat_len, feat_len]
                    # group_labels = tf.one_hot(groups_ids[:, i, :], tf.shape(group_logits)[1], dtype=tf.float32)
                    # [batch_size, feat_len]
                    group_shape = tf.shape(gids)
                    batch_idx = tf.expand_dims(tf.range(batch_size), 1)
                    # [batch_size, feat_len]
                    batch_idx = tf.tile(batch_idx, [1, group_shape[1]])
                    # [batch_size,feat_len, 1]
                    batch_idx = tf.expand_dims(batch_idx, 2)
                    # [batch_size, feat_len, 1]
                    feat_idx = tf.expand_dims(gids, 2)
                    # [batch_size, feat_len, 2]
                    feat_idx = tf.concat((batch_idx, feat_idx), 2)
                    # [batch_size, feat_cnt, hidden_size]
                    group_embed = tf.gather_nd(input_encoder_outputs, feat_idx)
                    group_mask = tf.sequence_mask(glens, tf.shape(group_embed)[1], dtype=tf.float32)
                    expanded_group_mask = tf.expand_dims(group_mask, 2)
                    # group_labels = tf.reduce_sum(group_labels * expanded_group_mask, 1)
                    expanded_glens = tf.expand_dims(glens, 1)
                    group_input = tf.reduce_sum(group_embed * expanded_group_mask, axis=1) / tf.to_float(
                        expanded_glens)
                    # [batch_size, m+1, feat_len]
                    infer_groups_ids = tf.concat((infer_groups_ids, tf.expand_dims(gids, 1)), 1)

                    return i + 1, group_input, group_state, infer_groups_ids, stop

                start_of_group_cell = tf.tile(init_input, [batch_size, 1])
                # gru的输入 batch_size, hidden*2
                group_state = group_init_state_fc(dec_input)
                group_state = tuple(group_state for _ in range(config.num_rnn_layers))
                # [batch_size, 1, feat_len] target [batch_size, group_len, feat_len]
                infer_groups_ids = tf.zeros((batch_size, 1, tf.shape(input_encoder_outputs)[1]), dtype=tf.int32)
                stop = tf.zeros((batch_size,), dtype=tf.int32)

                _, group_input, group_state, infer_groups_ids, glens, stop = tf.while_loop(group_cond,
                                                                                           group_body,
                                                                                           loop_vars=(
                                                                                               0,
                                                                                               start_of_group_cell,
                                                                                               group_state,
                                                                                               infer_groups_ids,
                                                                                               stop,
                                                                                               0))
                # 第0个变量是废弃的初始变量0000000
                infer_groups_ids = infer_groups_ids[:, 1:, :]

                infer_groups_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(infer_groups_ids, pad)), 2)
                infer_groups_cnt = tf.reduce_sum(tf.to_int32(tf.not_equal(infer_groups_lens, pad)), 1)
                infer_gidx, infer_group_bow, infer_group_mean_bow, infer_group_embed = self.gather_group(
                    input_encoder_outputs,
                    infer_groups_ids,
                    infer_groups_lens,
                    infer_groups_cnt,
                    group_encoder)

            with tf.variable_scope("infer_decoder"):
                # 共享train好的结构
                def infer_cond(i, predictions):
                    return i < tf.shape(infer_groups_ids)[1]

                def infer_body(i, predictions):
                    teacher_input_embed = groups_teacher_input_embed[:, i, :]
                    # batch_size, feat_len
                    teacher_input_ids = groups_teacher_input_ids[:, i, :]
                    teacher_input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(teacher_input_ids, pad)), 1)
                    sent_len = infer_groups_lens[:, i]
                    sent_bow = infer_group_bow[:, i, :, :]

                    tf.reduce_sum(tf.to_int32(tf.not_equal(key_input_ids, pad)), 1)
                    sent_output = outputs_ids[:, i, :]
                    with tf.name_scope("infer_sent_decoder"):
                        sent_dec_state = infer_group_mean_bow[:, i, :]
                        tile_encoder_state = tf.contrib.seq2seq.tile_batch(sent_dec_state, multiplier=beam_width)
                        tile_len = tf.contrib.seq2seq.tile_batch(sent_len, multiplier=beam_width)
                        tile_group = tf.contrib.seq2seq.tile_batch(sent_bow, multiplier=beam_width)
                        with tf.variable_scope("LuongAttention", reuse=True):
                            infer_attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=decoder_hidden_size,
                                                                                          memory=tile_group,
                                                                                          memory_sequence_length=tile_len)

                        infer_attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, infer_attention_mechanism,
                                                                              attention_layer_size=decoder_hidden_size)

                        decoder_initial_state = infer_attn_cell.zero_state(dtype=tf.float32,
                                                                           batch_size=batch_size * beam_width).clone(
                            cell_state=tile_encoder_state)

                        # length_penalty_weight=0.0, coverage_penalty_weight=0.0
                        beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(infer_attn_cell,
                                                                            word_embedding,
                                                                            start_tokens,
                                                                            pad,
                                                                            decoder_initial_state,
                                                                            beam_width=beam_width,
                                                                            output_layer=projection)

                        output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=beam_decoder,
                                                                         maximum_iterations=maximum_iterations)
                        # output.predicted_ids shape [batch_size, max_iterations, beam_width]
                        # [batch_size, max_iterations]
                        sent_output = tf.transpose(output.predicted_ids, [0, 2, 1])[:, 0, :]
                        # sent_output = sent_output * tf.to_int32(tf.greater_equal(sent_output, 0))
                        # dist = self.config.PHVM_maximum_iterations - tf.shape(sent_output)[1]
                        # padded_sent_output = tf.cond(tf.greater(dist, 0),
                        #                              lambda: tf.concat(
                        #                                  (sent_output,
                        #                                   tf.zeros((batch_size, dist), dtype=tf.int32)), 1),
                        #                              lambda: sent_output)
                        predictions = tf.concat((predictions, tf.expand_dims(sent_output, 1)), 1)
                        # sent_lens = tf.argmax(tf.cast(tf.equal(pad_output, 1), dtype=tf.int32), 1,
                        #                       output_type=tf.int32)
                        # sent_lens = sent_lens - 1 + tf.to_int32(tf.equal(sent_lens, 0)) * (tf.shape(sent_output)[1] + 1)
                        #
                        # output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=beam_decoder,
                        #                                                  maximum_iterations=maximum_iterations)
                        #
                    return i + 1, predictions

                predictions = tf.zeros((batch_size, 1, maximum_iterations), dtype=tf.int32)
                _, predictions = tf.while_loop(infer_cond,
                                               infer_body,
                                               loop_vars=(0, predictions))
                self.predictions = predictions

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

    def gather_group(self, feat_embed, group_ids, group_lens, group_cnt, group_encoder):
        shape = tf.shape(group_ids)
        batch_size = shape[0]
        fidx = tf.expand_dims(tf.expand_dims(tf.range(batch_size), 1), 2)
        fidx = tf.tile(fidx, [1, shape[1], shape[2]])
        fidx = tf.expand_dims(fidx, 3)
        sidx = tf.expand_dims(group_ids, 3)
        gidx = tf.concat((fidx, sidx), 3)
        group_bow = tf.gather_nd(feat_embed, gidx)
        group_mask = tf.sequence_mask(group_lens, shape[2], dtype=tf.float32)
        expanded_group_mask = tf.expand_dims(group_mask, 3)
        group_sum_bow = tf.reduce_sum(group_bow * expanded_group_mask, 2)
        safe_group_lens = group_lens + tf.cast(tf.equal(group_lens, 0), dtype=tf.int32)
        group_mean_bow = group_sum_bow / tf.to_float(tf.expand_dims(safe_group_lens, 2))

        group_encoder_output, group_encoder_state = tf.nn.dynamic_rnn(group_encoder,
                                                                      group_mean_bow,
                                                                      group_cnt,
                                                                      dtype=tf.float32)

        if self.config.PHVM_rnn_type == 'lstm':
            group_embed = group_encoder_state.h
        else:
            group_embed = group_encoder_state
        return gidx, group_bow, group_mean_bow, group_embed

    def select(self, group_probs, max_raw_feat_cnt):
        gids = []
        glens = []
        for probs in group_probs:
            tmp = []
            max_id = -1
            max_p = -1
            for feat_id, p in enumerate(probs):
                # 分类器阈值默认0.5
                if p >= self.config.group_selection_threshold:
                    tmp.append(feat_id)
                if p > max_p:
                    max_id = feat_id
                max_p = p
                # 如果说没有大于0.5的，就默认添加一个最大概率的feat到当前的分组
                if len(tmp) == 0:
                    tmp.append(max_id)
                gids.append(tmp)
                glens.append(len(tmp))
        # padding
        for item in gids:
            if len(item) < max_raw_feat_cnt:
                item += [0] * (max_raw_feat_cnt - len(item))
        return np.array(gids, dtype=np.int32), np.array(glens, dtype=np.int32)
