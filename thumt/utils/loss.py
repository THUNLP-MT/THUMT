# coding=utf-8
# Copyright 2018 The THUMT Authors

import tensorflow as tf
import thumt.utils.mrt_utils as mrt_utils


def get_loss(features, params, ce, tgt_mask):
    if params.use_mrt:
        loss = mrt_utils.mrt_loss(features, params, ce, tgt_mask)
    else:
        loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    return loss
