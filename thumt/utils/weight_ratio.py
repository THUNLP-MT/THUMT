# coding=utf-8
# Code modified from Tensor2Tensor library
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
import json
import math

from thumt.layers.nn import linear
from thumt.layers.attention import split_heads, combine_heads

def create_diagonal(output):
    '''
        output: (batchsize, dim)
        diagonal matrix (batchsize, length, length)
    '''
    length = tf.shape(output)[1]
    batchsize = tf.shape(output)[0]
    result = tf.diag(tf.ones([length]))
    result = tf.expand_dims(result, 0)
    result = tf.tile(result, [batchsize, 1, 1])
    return result


def weight_ratio_mean(input, output, stab=0):
    '''
        inputs: (..., dim)
        output: (..., 1)
        weight ratios: [(..., dim)]
    '''
    dim = tf.cast(tf.shape(input)[-1], tf.float32)
    output_shape = tf.shape(input)
    # Flatten to 2D
    inputs = tf.reshape(input, [-1, input.shape[-1].value])
    output = tf.reshape(output, [-1, output.shape[-1].value])

    w = inputs / dim / stabilize(output, stab)

    return tf.reshape(w, output_shape)


def stabilize(matrix, stab):
    sign = tf.sign(matrix)
    zero_pos = tf.equal(sign, tf.zeros(tf.shape(sign)))
    zero_pos = tf.cast(zero_pos, tf.float32)
    sign += zero_pos
    result = matrix + stab * sign
    return result


def weight_ratio_linear(inputs, weights, output, bias=None, stab=0):
    '''
        inputs: [(..., dim_in_i)]
        weights: [(dim_in_i, dim_out)]
        bias: [(dim_out)]
        output: (..., dim_out)
        weight ratios: [(..., dim_in_i, dim_out)]
    '''
    assert len(inputs) == len(weights)
    output_shape = []
    for i in range(len(inputs)):
        os = tf.concat([tf.shape(inputs[i]),tf.shape(weights[i])[-1:]],-1)
        output_shape.append(os)
    # Flatten to 2D
    inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]
    output = tf.reshape(output, [-1, output.shape[-1].value])

    weight_ratios = []

    for i in range(len(inputs)):
        r = tf.expand_dims(inputs[i],-1) * tf.expand_dims(weights[i], -3)
        w = r / tf.expand_dims(stabilize(output, stab), -2)
        weight_ratios.append(w)

    weight_ratios = [tf.reshape(wr, os)
                     for os, wr in zip(output_shape,weight_ratios)]

    return weight_ratios


def weight_ratio_weighted_sum(inputs, weights, output, stab=0, flatten=False):
    '''
        inputs: [(..., dim)]
        weights: [scalar]
        output: (..., dim)
        weight_ratios: [(..., dim, dim)]
    '''
    assert len(inputs) == len(weights)
    if flatten:
        output_shape = tf.shape(output)
    else:
        output_shape = tf.concat([tf.shape(output), tf.shape(output)[-1:]], -1)
    # Flatten to 2D
    inputs = [tf.reshape(inp, [-1, tf.shape(inp)[-1]]) for inp in inputs]
    output = tf.reshape(output, [-1, tf.shape(output)[-1]])

    weight_ratios = []
    diag = create_diagonal(output)
    for i in range(len(inputs)):
        wr = inputs[i] * weights[i] / stabilize(output, stab)
        if not flatten:
            wr = tf.expand_dims(wr, -1) * diag
        weight_ratios.append(wr)

    weight_ratios = [tf.reshape(wr, output_shape) for wr in weight_ratios]
    return weight_ratios


def weight_ratio_maxpool(input, output, maxnum, flatten=False):
    '''
        inputs: (..., dim)
        output: (..., dim/maxpart)
        weight_ratios: (..., dim, dim/maxnum)
    '''
    # Flatten to 2D
    maxnum = tf.constant(maxnum, dtype=tf.int32)
    weight_shape = tf.concat([tf.shape(input), tf.shape(output)[-1:]], axis=-1)
    input = tf.reshape(input, [-1, input.shape[-1].value])
    output = tf.reshape(output, [-1, output.shape[-1].value])

    shape_inp = tf.shape(input)
    batch = shape_inp[0]
    dim_in = shape_inp[-1]
    shape = tf.concat([shape_inp[:-1], [shape_inp[-1] // maxnum, maxnum]],
                      axis=0)
    dim_out = shape[-2]
    value = tf.reshape(input, shape)

    pos = tf.argmax(value, -1)
    pos = tf.cast(pos, tf.int32)
    pos = tf.reshape(pos, [-1])
    if flatten:
        indices = tf.range(tf.shape(pos)[0]) * maxnum + pos
        weight_ratio = tf.sparse_to_dense(indices, [batch * dim_in],
                                          tf.ones(tf.shape(indices)))
        weight_ratio = tf.reshape(weight_ratio, weight_shape[:-1])
    else:
        indices = dim_out * pos + dim_in * tf.range(batch * dim_out,
                                                    dtype=tf.int32)
        indices = tf.reshape(indices, [-1,dim_out])
        indices += tf.expand_dims(tf.range(dim_out, dtype=tf.int32), 0)
        indices = tf.reshape(indices, [-1])

        weight_ratio = tf.sparse_to_dense(indices, [batch * dim_in * dim_out],
                                          tf.ones(tf.shape(indices)))
        weight_ratio = tf.reshape(weight_ratio, weight_shape)

    return weight_ratio
