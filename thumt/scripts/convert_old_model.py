#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf


def parseargs():
    parser = argparse.ArgumentParser(description="Convert old models")

    parser.add_argument("--input", type=str, required=True,
                        help="Path of old model")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output checkpoint")

    return parser.parse_args()


def old_keys():
    keys = [
        "GRU_dec_attcontext",
        "GRU_dec_att",
        "GRU_dec_atthidden",
        "GRU_dec_inputoffset",
        "GRU_dec_inputemb",
        "GRU_dec_inputcontext",
        "GRU_dec_inputhidden",
        "GRU_dec_resetemb",
        "GRU_dec_resetcontext",
        "GRU_dec_resethidden",
        "GRU_dec_gateemb",
        "GRU_dec_gatecontext",
        "GRU_dec_gatehidden",
        "initer_b",
        "initer_W",
        "GRU_dec_probsemb",
        "GRU_enc_back_inputoffset",
        "GRU_enc_back_inputemb",
        "GRU_enc_back_inputhidden",
        "GRU_enc_back_resetemb",
        "GRU_enc_back_resethidden",
        "GRU_enc_back_gateemb",
        "GRU_enc_back_gatehidden",
        "GRU_enc_inputoffset",
        "GRU_enc_inputemb",
        "GRU_enc_inputhidden",
        "GRU_enc_resetemb",
        "GRU_enc_resethidden",
        "GRU_enc_gateemb",
        "GRU_enc_gatehidden",
        "GRU_dec_readoutoffset",
        "GRU_dec_readoutemb",
        "GRU_dec_readouthidden",
        "GRU_dec_readoutcontext",
        "GRU_dec_probsoffset",
        "GRU_dec_probs",
        "emb_src_b",
        "emb_src_emb",
        "emb_trg_b",
        "emb_trg_emb"
    ]

    return keys


def new_keys():
    keys = [
        "rnnsearch/decoder/attention/k_transform/matrix_0",
        "rnnsearch/decoder/attention/logits/matrix_0",
        "rnnsearch/decoder/attention/q_transform/matrix_0",
        "rnnsearch/decoder/gru_cell/candidate/bias",
        "rnnsearch/decoder/gru_cell/candidate/matrix_0",
        "rnnsearch/decoder/gru_cell/candidate/matrix_1",
        "rnnsearch/decoder/gru_cell/candidate/matrix_2",
        "rnnsearch/decoder/gru_cell/reset_gate/matrix_0",
        "rnnsearch/decoder/gru_cell/reset_gate/matrix_1",
        "rnnsearch/decoder/gru_cell/reset_gate/matrix_2",
        "rnnsearch/decoder/gru_cell/update_gate/matrix_0",
        "rnnsearch/decoder/gru_cell/update_gate/matrix_1",
        "rnnsearch/decoder/gru_cell/update_gate/matrix_2",
        "rnnsearch/decoder/s_transform/bias",
        "rnnsearch/decoder/s_transform/matrix_0",
        "rnnsearch/deepout/matrix_0",
        "rnnsearch/encoder/backward/gru_cell/candidate/bias",
        "rnnsearch/encoder/backward/gru_cell/candidate/matrix_0",
        "rnnsearch/encoder/backward/gru_cell/candidate/matrix_1",
        "rnnsearch/encoder/backward/gru_cell/reset_gate/matrix_0",
        "rnnsearch/encoder/backward/gru_cell/reset_gate/matrix_1",
        "rnnsearch/encoder/backward/gru_cell/update_gate/matrix_0",
        "rnnsearch/encoder/backward/gru_cell/update_gate/matrix_1",
        "rnnsearch/encoder/forward/gru_cell/candidate/bias",
        "rnnsearch/encoder/forward/gru_cell/candidate/matrix_0",
        "rnnsearch/encoder/forward/gru_cell/candidate/matrix_1",
        "rnnsearch/encoder/forward/gru_cell/reset_gate/matrix_0",
        "rnnsearch/encoder/forward/gru_cell/reset_gate/matrix_1",
        "rnnsearch/encoder/forward/gru_cell/update_gate/matrix_0",
        "rnnsearch/encoder/forward/gru_cell/update_gate/matrix_1",
        "rnnsearch/maxout/bias",
        "rnnsearch/maxout/matrix_0",
        "rnnsearch/maxout/matrix_1",
        "rnnsearch/maxout/matrix_2",
        "rnnsearch/softmax/bias",
        "rnnsearch/softmax/matrix_0",
        "rnnsearch/source_embedding/bias",
        "rnnsearch/source_embedding/embedding",
        "rnnsearch/target_embedding/bias",
        "rnnsearch/target_embedding/embedding",
    ]

    return keys


def main(args):
    values = dict(np.load(args.input))
    variables = {}
    o_keys = old_keys()
    n_keys = new_keys()

    for i, key in enumerate(o_keys):
        v = values[key]
        variables[n_keys[i]] = v

    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            tf_vars = [
                tf.get_variable(v, initializer=variables[v], dtype=tf.float32)
                for v in variables
            ]
            global_step = tf.Variable(0, name="global_step", trainable=False,
                                      dtype=tf.int64)

        saver = tf.train.Saver(tf_vars)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, args.output, global_step=global_step)


if __name__ == "__main__":
    main(parseargs())
