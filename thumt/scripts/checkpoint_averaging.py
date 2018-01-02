#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os

import numpy as np
import tensorflow as tf


def parseargs():
    msg = "Average checkpoints"
    usage = "average.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    parser.add_argument("--path", type=str, required=True,
                        help="checkpoint dir")
    parser.add_argument("--checkpoints", type=int, required=True,
                        help="number of checkpoints to use")
    parser.add_argument("--output", type=str, help="output path")

    return parser.parse_args()


def get_checkpoints(path):
    if not tf.gfile.Exists(os.path.join(path, "checkpoint")):
        raise ValueError("Cannot find checkpoints in %s" % path)

    checkpoint_names = []

    with tf.gfile.GFile(os.path.join(path, "checkpoint")) as fd:
        # Skip the first line
        fd.readline()
        for line in fd:
            name = line.strip().split(":")[-1].strip()[1:-1]
            key = int(name.split("-")[-1])
            checkpoint_names.append((key, os.path.join(path, name)))

    sorted_names = sorted(checkpoint_names, key=operator.itemgetter(0),
                          reverse=True)

    return [item[-1] for item in sorted_names]


def checkpoint_exists(path):
    return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
            tf.gfile.Exists(path + ".index"))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    checkpoints = get_checkpoints(FLAGS.path)
    checkpoints = checkpoints[:FLAGS.checkpoints]

    if not checkpoints:
        raise ValueError("No checkpoints provided for averaging.")

    checkpoints = [c for c in checkpoints if checkpoint_exists(c)]

    if not checkpoints:
        raise ValueError(
            "None of the provided checkpoints exist. %s" % FLAGS.checkpoints
        )

    var_list = tf.contrib.framework.list_variables(checkpoints[0])
    var_values, var_dtypes = {}, {}

    for (name, shape) in var_list:
        if not name.startswith("global_step"):
            var_values[name] = np.zeros(shape)

    for checkpoint in checkpoints:
        reader = tf.contrib.framework.load_checkpoint(checkpoint)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor
        tf.logging.info("Read from checkpoint %s", checkpoint)

    # Average checkpoints
    for name in var_values:
        var_values[name] /= len(checkpoints)

    tf_vars = [
        tf.get_variable(name, shape=var_values[name].shape,
                        dtype=var_dtypes[name]) for name in var_values
    ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step = tf.Variable(0, name="global_step", trainable=False,
                              dtype=tf.int64)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               var_values.iteritems()):
            sess.run(assign_op, {p: value})
        saved_name = os.path.join(FLAGS.output, "average")
        saver.save(sess, saved_name, global_step=global_step)

    tf.logging.info("Averaged checkpoints saved in %s", saved_name)

    params_pattern = os.path.join(FLAGS.path, "*.json")
    params_files = tf.gfile.Glob(params_pattern)

    for name in params_files:
        new_name = name.replace(FLAGS.path.rstrip("/"),
                                FLAGS.output.rstrip("/"))
        tf.gfile.Copy(name, new_name, overwrite=True)


if __name__ == "__main__":
    FLAGS = parseargs()
    tf.app.run()
