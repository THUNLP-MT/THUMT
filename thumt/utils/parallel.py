# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import operator

import tensorflow as tf


def _maybe_repeat(x, n):
    if isinstance(x, list):
        assert len(x) == n
        return x
    else:
        return [x] * n


# Data-level parallelism
def data_parallelism(devices, fn, *args, **kwargs):
    num_worker = len(devices)
    devices = ["gpu:%d" % d for d in devices]

    # Replicate args and kwargs
    if args:
        new_args = [_maybe_repeat(arg, num_worker) for arg in args]
        # Transpose
        new_args = [list(x) for x in zip(*new_args)]
    else:
        new_args = [[] for _ in range(num_worker)]

    new_kwargs = [{} for _ in range(num_worker)]

    for k, v in six.iteritems(kwargs):
        vals = _maybe_repeat(v, num_worker)

        for i in range(num_worker):
            new_kwargs[i][k] = vals[i]

    fns = _maybe_repeat(fn, num_worker)

    # Now make the parallel call.
    outputs = []

    for i in range(num_worker):
        with tf.variable_scope(tf.get_variable_scope(), reuse=(i != 0)):
            with tf.name_scope("parallel_%d" % i):
                with tf.device(devices[i]):
                    outputs.append(fns[i](*new_args[i], **new_kwargs[i]))

    return outputs


def shard_features(features, device_list):
    num_datashards = len(device_list)
    sharded_features = {}

    with tf.device("/cpu:0"):
        for k, v in six.iteritems(features):
            v = tf.convert_to_tensor(v)

            if not v.shape.as_list():
                v = tf.expand_dims(v, axis=-1)
                v = tf.tile(v, [num_datashards])

            batch_size = tf.shape(v)[0]
            size_splits = []

            for i in range(num_datashards):
                size_splits.append(
                    tf.cond(tf.greater(tf.mod(batch_size, num_datashards), i),
                            lambda: batch_size // num_datashards + 1,
                            lambda: batch_size // num_datashards)
                )

            sharded_features[k] = tf.split(v, size_splits, 0)

    datashard_to_features = []

    for d in range(num_datashards):
        feat = {
            k: v[d] for k, v in six.iteritems(sharded_features)
        }
        datashard_to_features.append(feat)

    return datashard_to_features


def parallel_model(model_fn, features, devices):
    if len(devices) == 1:
        return [model_fn(features)]

    features = shard_features(features, devices)

    outputs = data_parallelism(devices, model_fn, features)
    return outputs
