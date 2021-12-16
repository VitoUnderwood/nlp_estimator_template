# coding=utf-8
import collections
import re

import tensorflow as tf


def get_rnn_cell(rnn_type, num_layers, hidden_size, keep_prob, scope):
    """
    构建带有dropout的多层rnn结构，gru和lstm二选一
    """
    with tf.variable_scope(scope):
        lst = []
        for _ in range(num_layers):
            if rnn_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            else:
                cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            lst.append(cell)
        if num_layers > 1:
            res = tf.nn.rnn_cell.MultiRNNCell(lst)
        else:
            res = lst[0]
        return res


def get_out_put_from_tokens(all_sentences, rev_vocab):
    all_string_sent = []
    for each_sent in all_sentences:
        string_sent = []
        for each_word in each_sent:
            string_sent.append(rev_vocab.get(each_word))
        all_string_sent.append(' '.join(string_sent))
    return all_string_sent


def get_out_put_from_tokens_beam_search(all_sentences, id2word):
    all_string_sent = []
    for each_sent in all_sentences:
        each_sent = each_sent[:, 0]
        string_sent = []
        for each_word in each_sent:
            string_sent.append(id2word[each_word])
        all_string_sent.append(' '.join(string_sent))
    return all_string_sent


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names
