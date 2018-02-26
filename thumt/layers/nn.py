# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def linear(inputs, output_size, bias, concat=True, dtype=None, scope=None):
    """
    Linear layer
    :param inputs: A Tensor or a list of Tensors with shape [batch, input_size]
    :param output_size: An integer specify the output size
    :param bias: a boolean value indicate whether to use bias term
    :param concat: a boolean value indicate whether to concatenate all inputs
    :param dtype: an instance of tf.DType, the default value is ``tf.float32''
    :param scope: the scope of this layer, the default value is ``linear''
    :returns: a Tensor with shape [batch, output_size]
    :raises RuntimeError: raises ``RuntimeError'' when input sizes do not
                          compatible with each other
    """

    with tf.variable_scope(scope, default_name="linear", values=[inputs]):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)

            shape = [input_size, output_size]
            matrix = tf.get_variable("matrix", shape, dtype=dtype)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.get_variable("bias", shape, dtype=dtype)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)

        return output


def maxout(inputs, output_size, maxpart=2, use_bias=True, concat=True,
           dtype=None, scope=None):
    """
    Maxout layer
    :param inputs: see the corresponding description of ``linear''
    :param output_size: see the corresponding description of ``linear''
    :param maxpart: an integer, the default value is 2
    :param use_bias: a boolean value indicate whether to use bias term
    :param concat: concat all tensors if inputs is a list of tensors
    :param dtype: an optional instance of tf.Dtype
    :param scope: the scope of this layer, the default value is ``maxout''
    :returns: a Tensor with shape [batch, output_size]
    :raises RuntimeError: see the corresponding description of ``linear''
    """

    candidate = linear(inputs, output_size * maxpart, use_bias, concat,
                       dtype=dtype, scope=scope or "maxout")
    shape = tf.concat([tf.shape(candidate)[:-1], [output_size, maxpart]],
                      axis=0)
    value = tf.reshape(candidate, shape)
    output = tf.reduce_max(value, -1)

    return output


def layer_norm(inputs, epsilon=1e-6, dtype=None, scope=None):
    """
    Layer Normalization
    :param inputs: A Tensor of shape [..., channel_size]
    :param epsilon: A floating number
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string
    :returns: A Tensor with the same shape as inputs
    """
    with tf.variable_scope(scope, default_name="layer_norm", values=[inputs],
                           dtype=dtype):
        channel_size = inputs.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())

        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, -1, True)
        variance = tf.reduce_mean(tf.square(inputs - mean), -1, True)

        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset


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
                logits=logits,
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
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=soft_targets)

        if normalize is False:
            return xentropy

        # Normalizing constant is the best cross-entropy value with soft
        # targets. We subtract it just for readability, makes no difference on
        # learning
        normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

        return xentropy - normalizing
