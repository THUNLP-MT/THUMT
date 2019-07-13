# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def custom_getter(getter, name, shape=None, dtype=None, initializer=None,
                  regularizer=None, trainable=True, *args, **kwargs):
    var_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=var_dtype, initializer=initializer,
                      regularizer=regularizer, trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)

    return variable
