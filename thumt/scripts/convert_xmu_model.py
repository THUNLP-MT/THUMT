# coding=utf-8
# Copyright 2017 The THUMT Authors

import cPickle
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


def thu_keys():
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


def xmu_keys():
    keys = [
        "rnnsearch/decoder/attention/attention_w/matrix_0",
        "rnnsearch/decoder/attention/attention_v/matrix_0",
        "rnnsearch/decoder/attention/query_w/matrix_0",
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
        "rnnsearch/decoder/initial/bias",
        "rnnsearch/decoder/initial/matrix_0",
        "rnnsearch/decoder/deepout/matrix_0",
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
        "rnnsearch/decoder/maxout/bias",
        "rnnsearch/decoder/maxout/matrix_1",
        "rnnsearch/decoder/maxout/matrix_0",
        "rnnsearch/decoder/maxout/matrix_2",
        "rnnsearch/decoder/logits/bias",
        "rnnsearch/decoder/logits/matrix_0",
        "rnnsearch/source_embedding/bias",
        "rnnsearch/source_embedding/embedding",
        "rnnsearch/target_embedding/bias",
        "rnnsearch/target_embedding/embedding",
    ]

    return keys


def main(args):
    with open(args.input) as fd:
        _ = cPickle.load(fd)
        _ = cPickle.load(fd)
        values = dict(np.load(fd))

    variables = {}
    o_keys = xmu_keys()
    n_keys = thu_keys()

    for i, key in enumerate(o_keys):
        v = values[key]
        variables[n_keys[i]] = v

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
    arguments = parseargs()
    main(arguments)
