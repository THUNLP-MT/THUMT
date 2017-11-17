# coding=utf-8
# Copyright 2017 The THUMT Authors

import tensorflow as tf
import thumt.utils.mrt_utils as mrt_utils

def get_loss(features, params, ce, tgt_mask):
    if params.MRT:
        loss = mrt_utils.get_MRT_loss(features, params, ce, tgt_mask)
    else:
        loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    return loss
        
