# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def session_run(monitored_session, args):
    # Call raw TF session directly
    return monitored_session._tf_sess().run(args)


def zero_variables(variables, name=None):
    ops = []

    for var in variables:
        with tf.device(var.device):
            op = var.assign(tf.zeros(var.shape.as_list()))
        ops.append(op)

    return tf.group(*ops, name=name or "zero_op")


def replicate_variables(variables, device=None):
    new_vars = []

    for var in variables:
        device = device or var.device
        with tf.device(device):
            name = "replicate/" + var.name.split(":")[0]
            new_vars.append(tf.Variable(tf.zeros(var.shape.as_list()),
                                        name=name, trainable=False))

    return new_vars


def collect_gradients(gradients, variables):
    ops = []

    for grad, var in zip(gradients, variables):
        if isinstance(grad, tf.Tensor):
            ops.append(tf.assign_add(var, grad))
        else:
            ops.append(tf.scatter_add(var, grad.indices, grad.values))

    return tf.group(*ops)


def scale_gradients(gradients, scale):
    scaled_gradients = []

    for grad in gradients:
        if isinstance(grad, tf.IndexedSlices):
            slices = tf.IndexedSlices(scale * grad.values, grad.indices)
            scaled_gradients.append(slices)
        else:
            scaled_gradients.append(scale * grad)

    return tuple(scaled_gradients)
