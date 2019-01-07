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
import thumt.utils.weight_ratio as wr

import thumt.layers.nn as nn
import thumt.layers.attention as attention

from tensorflow.python.layers import base as base_layer

# Default value for INF
INF = 1. * 1e7

def reduce_reserve(matrix,axis):
    return tf.expand_dims(tf.reduce_sum(matrix, axis), axis)


def v2n_propagate_linear(w_x_last, w_linear):
    '''
        w_x_last: [bs, len_src, len, d]
        w_linear: [b, len, d, d]
        return: [b, len_src, len, d]
    '''
    len_src = tf.shape(w_x_last)[1]
    w_x_last = tf.expand_dims(w_x_last, -2)
    w_linear = tf.expand_dims(w_linear, 1)
    w_linear = tf.tile(w_linear, [1, len_src, 1, 1, 1])
    result = tf.matmul(w_x_last, w_linear)
    result = tf.squeeze(result, -2)
    return result


def create_diagonal_v2n(batchsize, length, dim):
    '''
        diagonal matrix (batchsize, len_src, len_src, dim)
	result[bs, len_src, len_src, dim] = 1
    '''
    result = tf.diag(tf.ones([length], dtype=tf.float32))
    result = tf.expand_dims(result, 0)
    result = tf.expand_dims(result, -1)
    result = tf.tile(result, [batchsize, 1, 1, dim])
    return result


def stabilize(matrix, stab):
    sign = tf.sign(matrix)
    zero_pos = tf.equal(sign, tf.zeros(tf.shape(sign)))
    zero_pos = tf.cast(zero_pos, tf.float32)
    sign += zero_pos
    result = matrix + stab * sign
    return result


def normalize(matrix):
    total = tf.reduce_sum(tf.abs(matrix), axis=-1)
    return matrix / tf.expand_dims(total, axis=-1)


def dot_product(inputs, params):
    output = tf.identity(inputs[0])
    for i in range(1, len(inputs)):
        output *= inputs[i]

    weight_ratios = wr.weight_ratio_dot_product(inputs, output)

    return {"output": output, "weight_ratios": weight_ratios}


def weighted_sum(inputs, weights, params, flatten=False):
    assert len(inputs) == len(weights)
    output = tf.add_n([inputs[i] * weights[i] for i in range(len(inputs))])

    weight_ratios = wr.weight_ratio_weighted_sum(inputs, weights, output,
                                                 stab=params.stab,
                                                 flatten=flatten)

    return {"output": output, "weight_ratios": weight_ratios}


def maxpool(input, output_size, params, flatten=False):
    shape = tf.concat([tf.shape(input)[:-1], [output_size, params.maxnum]],
                      axis=0)
    value = tf.reshape(input, shape)
    output = tf.reduce_max(value, -1)
    weight_ratio = wr.weight_ratio_maxpool(input, output, params.maxnum,
                                           flatten=flatten)
    return {"output": output, "weight_ratio": weight_ratio}


def weight_ratio_linear_v2n_2d(inputs, weights, output, w_x_inp, bias=None,
                               stab=0):
    inputs_ex = [tf.expand_dims(inp, 1) for inp in inputs]
    output_ex = tf.expand_dims(output, 1)
    w_x_inp_ex = [tf.expand_dims(w, 2) for w in w_x_inp]
    result = weight_ratio_linear_v2n(inputs_ex, weights, output_ex,
                                    w_x_inp_ex, bias=bias, stab=stab)
    result = [tf.squeeze(res, 2) for res in result]
    return result


def weight_ratio_linear_v2n(inputs, weights, output, w_x_inp, bias=None,
                            stab=0):
    '''
        inputs: [(bs, lq, di)]
        weights: [(di, do)]
        bias: (do)
        output: (bs, lq, do)
        w_x_inp: [(bs, ls, lq, di)]
        weight ratios: [(bs, ls, lq, do)]
    '''
    assert len(inputs) == len(weights)
    weight_ratios = []
    bs = tf.shape(w_x_inp[0])[0]
    lq = tf.shape(w_x_inp[0])[2]
    outp = tf.expand_dims(stabilize(output, stab), 1)
    outp = tf.reshape(outp, [bs,1,lq,-1])

    for i in range(len(inputs)):
        di = tf.shape(w_x_inp[i])[3]
        ls = tf.shape(w_x_inp[i])[1]
        inp = tf.reshape(inputs[i], [bs,1,lq,-1])
        w = w_x_inp[i] * inp
        w = tf.reshape(w, [-1, di])
        w = tf.matmul(w, weights[i])
        w = tf.reshape(w, [bs, ls, lq, -1])
        w = w / outp
        weight_ratios.append(w)

    return weight_ratios


def linear_v2n(inputs, output_size, bias, w_x_inp, params, concat=False,
               dtype=None, scope=None, d2=False):
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
        #assert not concat
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        batch_shape = tf.shape(inputs[0])[:-1]
        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []
        weight_ratios = []
        weight_shapes = []
        matrixs = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)
            shape = [input_size, output_size]
            weight_shape = tf.concat([batch_shape,shape], -1)
            matrix = tf.get_variable("matrix", shape, dtype=dtype)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                weight_shapes.append(tf.concat([batch_shape, shape], -1))
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype)
                matrixs.append(matrix)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.get_variable("bias", shape, dtype=dtype)
            output = tf.nn.bias_add(output, bias)

        # calculate weight ratio
        operator = weight_ratio_linear_v2n
        if d2:
            operator = weight_ratio_linear_v2n_2d
        if concat:
            weight_ratios = operator([inputs], [matrix], output, w_x_inp,
                                     bias=bias, stab=params.stab)
        else:
            weight_ratios = operator(inputs, matrixs, output, w_x_inp,
                                     bias=bias, stab=params.stab)

        output = tf.reshape(output, output_shape)

        return {"output":output, "weight_ratios": weight_ratios}


def maxout_v2n(inputs, output_size, maxpart, w, params, use_bias=True,
               concat=True, dtype=None, scope=None):
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

    w_x_dec, w_x_ctx = w
    w_x_dec = tf.transpose(w_x_dec, [1, 2, 0, 3])
    w_x_ctx = tf.transpose(w_x_ctx, [1, 2, 0, 3])
    w_x_y = tf.zeros(tf.shape(w_x_dec), dtype=tf.float32)
    candidate_linear = linear_v2n(inputs, output_size * maxpart, use_bias,
                                  [w_x_y, w_x_dec, w_x_ctx], params, concat,
                                  dtype=dtype, scope=scope or "maxout")
    candidate = candidate_linear["output"]
    _, w_x_dec_readout, w_x_ctx_readout = candidate_linear["weight_ratios"]
    w_x_readout = w_x_dec_readout + w_x_ctx_readout
    w_x_readout = tf.transpose(w_x_readout, [0, 2, 1, 3])

    output_maxout = maxpool(candidate, output_size, params)
    output = output_maxout["output"]

    # direct
    w_readout_maxout = output_maxout["weight_ratio"]

    #propagate
    propagater = tf.matmul

    w_x_maxout = propagater(w_x_readout, w_readout_maxout)

    weight_ratios = [w_x_maxout]

    return {"output": output, "weight_ratios": weight_ratios}

class LegacyGRUCell_encoder_v2n(tf.nn.rnn_cell.RNNCell):
    """ Groundhog's implementation of GRUCell

    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None):
        super(LegacyGRUCell_encoder_v2n, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, state, w_x_h_last, params, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            bs = tf.shape(w_x_h_last)[0]
            emb = tf.shape(inputs)[-1]
            w_x_x = tf.ones([bs,1,emb], dtype=tf.float32)
            all_inputs = list(inputs) + [state]
            r_linear = linear_v2n(all_inputs, self._num_units, False,
                                  [w_x_x, w_x_h_last], params, False,
                                  scope="reset_gate", d2=True)
            w_x_r, w_xlast_r = r_linear["weight_ratios"]
            r = tf.nn.sigmoid(r_linear["output"])
            u_linear = linear_v2n(all_inputs, self._num_units, False,
                                  [w_x_x, w_x_h_last], params, False,
                                  scope="update_gate", d2=True)
            w_x_u, w_xlast_u = u_linear["weight_ratios"]
            u = tf.nn.sigmoid(u_linear["output"])

            reseted = r * state
            w_x_reseted = w_x_r
            w_xlast_reseted = w_xlast_r

            w_tx_reseted = tf.concat([w_x_reseted, w_xlast_reseted], 1)
            all_inputs = list(inputs) + [reseted]
            c_linear = linear_v2n(all_inputs, self._num_units, True,
                                  [w_x_x, w_tx_reseted], params, False,
                                  scope="candidate", d2=True)
            w_x_c_direct, w_tx_reseted_c = c_linear["weight_ratios"]
            w_x_reseted_c, w_xlast_c = tf.split(w_tx_reseted_c,
                                        [1, tf.shape(w_tx_reseted_c)[1]-1],
                                        axis=1)
            w_x_c = w_x_c_direct + w_x_reseted_c
            c = c_linear["output"]

            h1 = u * tf.tanh(c)
            h2 = (1.0 - u) * state
            new_state = h1 + h2
            new_state_stab = stabilize(new_state, params.stab)
            w_x_newh = w_x_c * tf.expand_dims(h1/new_state_stab, axis=1)
            w_xlast_newh = \
                      w_xlast_c * tf.expand_dims(h1/new_state_stab, axis=1) + \
                      w_x_h_last * tf.expand_dims(h2/new_state_stab, axis=1)

        return new_state, new_state, w_xlast_newh, w_x_newh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

class LegacyGRUCell_decoder_v2n(tf.nn.rnn_cell.RNNCell):
    """ Groundhog's implementation of GRUCell

    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None):
        super(LegacyGRUCell_decoder_v2n, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, state, w_x_h_last, w_x_c, params, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            bs = tf.shape(w_x_h_last)[0]
            emb = tf.shape(inputs[0])[-1]
            w_x_y = tf.zeros([bs, 1, emb], dtype=tf.float32)
            all_inputs = list(inputs) + [state]
            w_x_h = w_x_h_last
            w_x_ctx = w_x_c
            r_linear = linear_v2n(all_inputs, self._num_units, False,
                                  [w_x_y, w_x_c, w_x_h_last], params, False,
                                  scope="reset_gate", d2=True)
            _, w_x_ctx_r, w_x_h_r = r_linear["weight_ratios"]
            w_x_r = w_x_ctx_r + w_x_h_r
            r = tf.nn.sigmoid(r_linear["output"])

            u_linear = linear_v2n(all_inputs, self._num_units, False,
                                  [w_x_y, w_x_c, w_x_h_last], params, False,
                                  scope="update_gate", d2=True)
            _, w_x_ctx_u, w_x_h_u = u_linear["weight_ratios"]
            w_x_u = w_x_ctx_u + w_x_h_u
            u = tf.nn.sigmoid(u_linear["output"])

            reseted = r * state
            w_x_reseted = 0.5 * w_x_r + 0.5 * w_x_h

            all_inputs = list(inputs) + [reseted]
            c_linear = linear_v2n(all_inputs, self._num_units, True,
                                  [w_x_y, w_x_c, w_x_reseted], params, False,
                                  scope="candidate", d2=True)
            _, w_x_c_state, w_x_resetes_state = c_linear["weight_ratios"]
            w_x_state = w_x_c_state + w_x_resetes_state
            c = c_linear["output"]

            h1 = u * tf.tanh(c)
            h2 = (1.0 - u) * state
            w_x_h1 = 0.5 * w_x_state + 0.5 * w_x_u
            w_x_h2 = 0.5 * w_x_u + 0.5 * w_x_h

            newh_ws = weighted_sum([h1, h2], [1., 1.], params, flatten=True)
            new_state = newh_ws["output"]
            w_h1_newh, w_h2_newh = newh_ws["weight_ratios"]
            w_h1_newh = tf.expand_dims(w_h1_newh, 1)
            w_h2_newh = tf.expand_dims(w_h2_newh, 1)
            w_x_newh = w_x_h1 * w_h1_newh + w_x_h2 * w_h2_newh

        return new_state, new_state, w_x_newh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


def combine_heads_v2n(inputs, name=None):
    with tf.name_scope(name, default_name="combine_heads", values=[inputs]):
        x = inputs
        x = tf.transpose(x, [0, 2, 3, 1, 4])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)

        return x


def split_heads(inputs, num_heads, name=None):
    with tf.name_scope(name, default_name="split_heads", values=[inputs]):
        x = inputs
        n = num_heads
        old_shape = x.get_shape().dims

        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret, [0, 3, 1, 2, 4])

def combine_heads(inputs, name=None):
    with tf.name_scope(name, default_name="combine_heads", values=[inputs]):
        x = inputs
        x = tf.transpose(x, [0, 2, 3, 4, 1, 5])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)

        return x


def layer_process(x, mode, w_x_inp, params):
    if not mode or mode == "none":
        return {"outputs": x, "weight_ratios": w_x_inp}
    elif mode == "layer_norm":
        norm = layer_norm(x, w_x_inp, params)
        return norm
    else:
        raise ValueError("Unknown mode %s" % mode)


def layer_norm(inputs, w_x_inp, params, epsilon=1e-6, dtype=None, scope=None):
    """
    Layer Normalization
    :param inputs: A Tensor of shape [..., channel_size]
    :param epsilon: A floating number
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string
    :returns: A Tensor with the same shape as inputs

    w_x_inp: [bs, len_src, len, dim]
    """
    with tf.variable_scope(scope, default_name="layer_norm", values=[inputs],
                           dtype=dtype):
        channel_size = inputs.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())

        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1,
                                  keep_dims=True)

        averaged = (inputs - mean)
        norm_inputs = averaged * tf.rsqrt(variance + epsilon)

        w_inp_mean = wr.weight_ratio_mean(inputs, mean, stab=params.stab)
        w_inp_out, w_mean_out = wr.weight_ratio_weighted_sum([inputs, mean],
                                                             [1.,-1.],
                                                             averaged,
                                                             stab=params.stab,
                                                             flatten=True)
        w_x_mean = tf.reduce_sum(w_x_inp * tf.expand_dims(w_inp_mean, 1), -1)
        w_inp_out = tf.expand_dims(w_inp_out, 1)
        w_mean_out = tf.expand_dims(w_mean_out, 1)
        w_x_out = w_x_inp * w_inp_out
        w_x_out += tf.expand_dims(w_x_mean, -1) * w_mean_out

        return {"outputs":norm_inputs * scale + offset,
                "weight_ratios": w_x_out}

def multihead_attention_v2n(queries, memories, bias, w_x_inp, num_heads,
                            key_size, value_size, output_size, params,
                            keep_prob=None, output=True, dtype=None,
                            scope=None):
    """ Multi-head scaled-dot-product attention with input/output
        transformations.

    :param queries: A tensor with shape [batch, length_q, depth_q] if
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param keep_prob: A floating point number in (0, 1]
    :param output: Whether to use output transformation
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string


    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_v]
        outputs: A tensor with shape [batch, length_q, depth_v]
        weight_ratio: [batch. length_q, d, length_v, d]

        w_x_inp: [batch, len_src, len_src, d] or [batch, len_trg, len_trg, d]
    """

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_attention",
                           values=[queries, memories], dtype=dtype):
        bs = tf.shape(w_x_inp)[0]
        len_q = tf.shape(queries)[1]
        len_src = tf.shape(w_x_inp)[1]
        dim = tf.shape(w_x_inp)[3]
        if memories is None:
            # self attention
            size = key_size * 2 + value_size
            combined_linear = linear_v2n(queries, size, True, [w_x_inp],
                                         params, True, scope="qkv_transform")
            combined = combined_linear["output"]
            q, k, v = tf.split(combined, [key_size, key_size, value_size],
                               axis=-1)
            w_x_combined = combined_linear["weight_ratios"][0]
            w_x_q, w_x_k, w_x_v = tf.split(w_x_combined,
                                           [key_size, key_size, value_size],
                                           axis=-1)
        else:
            q = nn.linear(queries, key_size, True, params, True,
                          scope="q_transform")
            combined_linear = linear_v2n(memories, key_size + value_size, True,
                                         [w_x_inp], params, True,
                                         scope="kv_transform")
            combined = combined_linear["output"]
            k, v = tf.split(combined, [key_size, value_size], axis=-1)
            w_x_combined = combined_linear["weight_ratios"][0]
            w_x_k, w_x_v = tf.split(w_x_combined, [key_size, value_size],
                                    axis=-1)

        # split heads
        q = attention.split_heads(q, num_heads)
        k = attention.split_heads(k, num_heads)
        v = attention.split_heads(v, num_heads)
        w_x_v = split_heads(w_x_v, num_heads)

        # scale query
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        # attention
        results = attention.multiplicative_attention(q, k, v, bias, keep_prob)

        # combine heads
        weights = results["weights"]
        x = attention.combine_heads(results["outputs"])

        w_x_v = tf.transpose(w_x_v, [0, 1, 3, 2, 4])
        w_x_v = tf.reshape(w_x_v, [bs, num_heads, tf.shape(w_x_v)[2], -1])
        w_x_att = tf.matmul(weights, w_x_v)
        w_x_att = tf.reshape(w_x_att,
                        [bs, num_heads, len_q, len_src, key_depth_per_head])
        w_x_att = tf.transpose(w_x_att, [0, 1, 3, 2, 4])
        w_x_att = combine_heads_v2n(w_x_att)

        if output:
            outputs_linear = linear_v2n(x, output_size, True, [w_x_att],
                                        params, True,
                                        scope="output_transform")
            outputs = outputs_linear["output"]
            w_x_out = outputs_linear["weight_ratios"][0]
        else:
            outputs = x
            w_x_out = w_x_att

        return {"weights": weights, "outputs": outputs, "weight_ratio": w_x_out}

