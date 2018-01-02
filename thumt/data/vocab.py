# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def load_vocabulary(filename):
    vocab = []
    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            word = line.strip()
            vocab.append(word)

    return vocab


def process_vocabulary(vocab, params):
    if params.append_eos:
        vocab.append(params.eos)

    return vocab


def get_control_mapping(vocab, symbols):
    mapping = {}

    for i, token in enumerate(vocab):
        for symbol in symbols:
            if symbol.decode("utf-8") == token.decode("utf-8"):
                mapping[symbol] = i

    return mapping
