# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def smoothed_softmax_cross_entropy_with_logits(**kwargs):
    logits = kwargs.get("logits")
    labels = kwargs.get("labels")
    smoothing = kwargs.get("smoothing") or 0.0
    normalize = kwargs.get("normalize")
    scope = kwargs.get("scope")

    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    with tf.name_scope(scope or "smoothed_softmax_cross_entropy_with_logits",
                       values=[logits, labels]):

        labels = tf.reshape(labels, [-1])

        if not smoothing:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.cast(logits, tf.float32),
                labels=labels
            )
            return ce

        # label smoothing
        vocab_size = tf.shape(logits)[1]

        n = tf.to_float(vocab_size - 1)
        p = 1.0 - smoothing
        q = smoothing / n

        soft_targets = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size,
                                  on_value=p, off_value=q)
        soft_targets = tf.stop_gradient(soft_targets)
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=tf.cast(logits, tf.float32),
            labels=soft_targets)

        if normalize is False:
            return xentropy

        # Normalizing constant is the best cross-entropy value with soft
        # targets. We subtract it just for readability, makes no difference on
        # learning
        normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

        return xentropy - normalizing
