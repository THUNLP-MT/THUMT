# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cache_features(features, num_shards):
    if num_shards == 1:
        return features, tf.no_op(name="init_queue")

    flat_features = list(features.itervalues())
    queue = tf.FIFOQueue(num_shards, dtypes=[v.dtype for v in flat_features])
    flat_features = [tf.split(v, num_shards, axis=0) for v in flat_features]
    flat_features = list(zip(*flat_features))
    init_ops = [queue.enqueue(v, name="enqueue_%d" % i)
                for i, v in enumerate(flat_features)]
    flat_feature = queue.dequeue()
    new_features = {}

    for k, v in zip(features.iterkeys(), flat_feature):
        v.set_shape(features[k].shape)
        new_features[k] = v

    return new_features, tf.group(*init_ops)
