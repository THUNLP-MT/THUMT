# coding=utf-8
# Copyright 2017 The THUMT Authors

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
        if token in symbols:
            mapping[token] = i

    return mapping
