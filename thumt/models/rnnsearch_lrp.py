# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils.lrp as lrp

from thumt.models.model import NMTModel


def normalize(matrix, negative=False):
    if negative:
        matrix_abs = tf.abs(matrix)
        total = tf.reduce_sum(matrix_abs, -1)
        return matrix / tf.expand_dims(total, -1)
    else:
        matrix = tf.abs(matrix)
        total = tf.reduce_sum(matrix, -1)
        return matrix / tf.expand_dims(total, -1)


def stabilize(matrix, stab):
    sign = tf.sign(matrix)
    zero_pos = tf.equal(sign, tf.zeros(tf.shape(sign)))
    zero_pos = tf.cast(zero_pos, tf.float32)
    sign += zero_pos
    result = matrix + stab * sign
    return result


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)


def _gru_encoder(cell, inputs, sequence_length, initial_state, params,
                 dtype=None):
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")

    w_x_h_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="w_x_h_array")
    w_x_h_init = tf.zeros([batch, time_steps, output_size], dtype=tf.float32)

    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state, wxh_ta, w_x_h_last):
        inp_t = input_ta.read(t)
        cell_output, new_state, w_xlast_newh, w_x_newh = cell(inp_t, state,
                                                              w_x_h_last,
                                                              params)
        w_x_newh = tf.pad(w_x_newh, [[0,0], [t, time_steps - t - 1], [0,0]])
        w_x_h_new = w_xlast_newh + w_x_newh
        cell_output = _copy_through(t, sequence_length, zero_output,
                                    cell_output)
        new_state = _copy_through(t, sequence_length, state, new_state)
        w_x_h_new = _copy_through(t, sequence_length, w_x_h_last, w_x_h_new)
        out_ta = out_ta.write(t, cell_output)
        wxh_ta = wxh_ta.write(t, w_x_h_new)
        return t + 1, out_ta, new_state, wxh_ta, w_x_h_new

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state, w_x_h_ta, w_x_h_init)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    w_x_h_final_ta = outputs[3]
    w_x_h_final = w_x_h_final_ta.stack()

    return all_output, final_state, w_x_h_final


def _encoder(cell_fw, cell_bw, inputs, sequence_length, params, dtype=None,
             scope=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length]):
        inputs_fw = inputs
        inputs_bw = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)

        with tf.variable_scope("forward"):
            output_fw, state_fw, w_x_h_fw = _gru_encoder(cell_fw, inputs_fw,
                                               sequence_length, None, params,
                                               dtype=dtype)

        with tf.variable_scope("backward"):
            output_bw, state_bw, w_x_h_bw = _gru_encoder(cell_bw, inputs_bw,
                                               sequence_length, None, params,
                                               dtype=dtype)
            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            batch_axis=0, seq_axis=1)

        results = {
            "annotation": tf.concat([output_fw, output_bw], axis=2),
            "outputs": {
                "forward": output_fw,
                "backward": output_bw
            },
            "final_states": {
                "forward": state_fw,
                "backward": state_bw
            },
            "weight_ratios": [w_x_h_fw, w_x_h_bw]
        }

        return results


def _decoder(cell, inputs, memory, sequence_length, initial_state, w_x_enc,
             w_x_bw, params, dtype=None, scope=None):
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])
        mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(memory)[1],
                                    dtype=tf.float32)
        bias = layers.attention.attention_bias(mem_mask, "masking")
        bias = tf.squeeze(bias, axis=[1, 2])
        cache = layers.attention.attention(None, memory, None, output_size)

        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
        value_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="value_array")
        alpha_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="alpha_array")
        input_ta = input_ta.unstack(inputs)

        len_src = tf.shape(w_x_bw)[0]
        w_x_bw_ta = tf.TensorArray(tf.float32, len_src,
                                   tensor_array_name="w_x_bw_array")

        w_x_bw_ta = w_x_bw_ta.unstack(w_x_bw)
        w_x_c_shape = tf.shape(w_x_enc)[1:]
        w_x_enc = tf.transpose(w_x_enc, [1,0,2,3])
        w_x_enc = tf.reshape(w_x_enc,
                             tf.concat([tf.shape(w_x_enc)[:2], [-1]], -1))

        w_x_h_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="w_x_h_array")
        w_x_ctx_ta = tf.TensorArray(tf.float32, time_steps,
                                    tensor_array_name="w_x_ctx_array")

        initial_state_linear = lrp.linear_v2n(initial_state, output_size, True,
                                              [w_x_bw_ta.read(0)], params,
                                              False, scope="s_transform",
                                              d2=True)
        initial_state = initial_state_linear["output"]
        w_initial = initial_state_linear["weight_ratios"][0]
        initial_state = tf.tanh(initial_state)

        def loop_func(t, out_ta, att_ta, val_ta, state, cache_key, wxh_ta,
                      wxc_ta, w_x_h_last):
            # now state
            wxh_ta = wxh_ta.write(t, w_x_h_last)
            inp_t = input_ta.read(t)
            results = layers.attention.attention(state, memory, bias,
                                                 output_size,
                                                 cache={"key": cache_key})
            alpha = results["weight"]
            context = results["value"]

            att = tf.expand_dims(alpha, 1)

            wr_att = tf.expand_dims(att, -1) * tf.expand_dims(memory, 1)
            wr_att = tf.squeeze(wr_att, 1)
            result_stab = stabilize(context, params.stab)
            wr_att /= tf.expand_dims(result_stab, 1)
            len_src = tf.shape(wr_att)[1]
            w_x_c = tf.reshape(w_x_enc, [1, len_src, len_src, -1]) * wr_att
            w_x_c = tf.reduce_sum(w_x_c, 2)

            #w_x_c = tf.matmul(att, w_x_enc)
            w_x_c = tf.reshape(w_x_c, w_x_c_shape)
            wxc_ta = wxc_ta.write(t, w_x_c)

            # next state
            cell_input = [inp_t, context]
            cell_output, new_state, w_x_h_new = cell(cell_input, state,
                                                     w_x_h_last, w_x_c, params)
            cell_output = _copy_through(t, sequence_length["target"],
                                        zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state,
                                      new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value,
                                      context)
            w_x_h_new = _copy_through(t, sequence_length["target"], w_x_h_last,
                                      w_x_h_new)

            out_ta = out_ta.write(t, cell_output)
            att_ta = att_ta.write(t, alpha)
            val_ta = val_ta.write(t, new_value)
            cache_key = tf.identity(cache_key)

            return t + 1, out_ta, att_ta, val_ta, new_state, cache_key, \
                   wxh_ta, wxc_ta, w_x_h_new

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, alpha_ta, value_ta, initial_state,
                     cache["key"], w_x_h_ta, w_x_ctx_ta, w_initial)

        outputs = tf.while_loop(lambda t, *_: t < time_steps,
                                loop_func, loop_vars,
                                parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        value_final_ta = outputs[3]

        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])

        final_value = value_final_ta.stack()
        final_value.set_shape([None, None, memory.shape[-1].value])
        final_value = tf.transpose(final_value, [1, 0, 2])

        w_x_h_final_ta = outputs[6]
        w_x_h_final = w_x_h_final_ta.stack()
        w_x_c_final_ta = outputs[7]
        w_x_c_final = w_x_c_final_ta.stack()

        result = {
            "outputs": final_output,
            "values": final_value,
            "initial_state": initial_state,
            "weight_ratios": [w_x_h_final, w_x_c_final, w_initial]
        }

    return result


def model_graph(features, labels, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])

    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding",
                                  [src_vocab_size, params.embedding_size])
        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

    with tf.variable_scope("target_embedding"):
        tgt_emb = tf.get_variable("embedding",
                                  [tgt_vocab_size, params.embedding_size])
        tgt_bias = tf.get_variable("bias", [params.embedding_size])
        tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])

    src_inputs = tf.nn.bias_add(src_inputs, src_bias)
    tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)

    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
        tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

    # encoder
    cell_fw = lrp.LegacyGRUCell_encoder_v2n(params.hidden_size)
    cell_bw = lrp.LegacyGRUCell_encoder_v2n(params.hidden_size)

    if params.use_variational_dropout:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )

    encoder_output = _encoder(cell_fw, cell_bw, src_inputs,
                              features["source_length"], params)

    w_x_h_fw, w_x_h_bw = encoder_output["weight_ratios"]
    w_x_h_bw = w_x_h_bw[::-1, :, ::-1]
    w_x_enc = tf.concat([w_x_h_fw, w_x_h_bw], -1)

    # decoder
    cell = lrp.LegacyGRUCell_decoder_v2n(params.hidden_size)

    if params.use_variational_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            # input + context
            input_size=params.embedding_size + 2 * params.hidden_size,
            dtype=tf.float32
        )

    length = {
        "source": features["source_length"],
        "target": features["target_length"]
    }
    initial_state = encoder_output["final_states"]["backward"]
    decoder_output = _decoder(cell, tgt_inputs, encoder_output["annotation"],
                              length, initial_state, w_x_enc, w_x_h_bw, params)

    w_x_dec, w_x_ctx, w_x_init = decoder_output["weight_ratios"]

    # Shift left
    shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
    shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]

    all_outputs = tf.concat(
        [
            tf.expand_dims(decoder_output["initial_state"], axis=1),
            decoder_output["outputs"],
        ],
        axis=1
    )
    shifted_outputs = all_outputs[:, :-1, :]

    maxout_features = [
        shifted_tgt_inputs,
        shifted_outputs,
        decoder_output["values"]
    ]
    maxout_size = params.hidden_size // params.maxnum

    if labels is None:
        # Special case for non-incremental decoding
        maxout_features = [
            shifted_tgt_inputs[:, -1, :],
            shifted_outputs[:, -1, :],
            decoder_output["values"][:, -1, :]
        ]
        maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                              params, concat=False)

        readout = layers.nn.linear(maxhid, params.embedding_size, False,
                                   False, scope="deepout")

        # Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True, False,
                                  scope="softmax")

        return logits

    maxhid_maxout = lrp.maxout_v2n(maxout_features, maxout_size, params.maxnum,
                                   [w_x_dec, w_x_ctx], params, concat=False)
    maxhid = maxhid_maxout["output"]
    w_x_maxout = maxhid_maxout["weight_ratios"][0]
    w_x_maxout = tf.transpose(w_x_maxout, [0, 2, 1, 3])
    readout = lrp.linear_v2n(maxhid, params.embedding_size, False,
                             [w_x_maxout], params, False, scope="deepout")
    w_x_readout = readout["weight_ratios"][0]
    readout = readout["output"]

    if params.dropout and not params.use_variational_dropout:
        readout = tf.nn.dropout(readout, 1.0 - params.dropout)

    # Prediction and final relevance
    logits = lrp.linear_v2n(readout, tgt_vocab_size, True, [w_x_readout],
                            params, False, scope="softmax")
    w_x_true = logits["weight_ratios"][0]
    logits= logits["output"]
    logits = tf.reshape(logits, [-1, tgt_vocab_size])
    w_x_true = tf.transpose(w_x_true, [0, 2, 1, 3])
    w_x_true = tf.reshape(w_x_true, [-1, tf.shape(w_x_true)[-2],
                                     tf.shape(w_x_true)[-1]])
    w_x_true = tf.transpose(w_x_true, [0, 2, 1])
    labels_lrp = labels
    bs = tf.shape(labels_lrp)[0]
    idx = tf.range(tf.shape(labels_lrp)[-1])
    idx = tf.cast(idx, tf.int64)
    idx = tf.reshape(idx, [1, -1])
    labels_lrp = tf.concat([idx, labels_lrp], axis=0)
    labels_lrp = tf.transpose(labels_lrp, [1, 0])
    w_x_true = tf.gather_nd(w_x_true, labels_lrp)
    w_x_true = tf.reshape(w_x_true, [bs, -1, tf.shape(w_x_true)[-1]])

    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(labels))
    tgt_mask = tf.to_float(
        tf.sequence_mask(
            features["target_length"],
            maxlen=tf.shape(features["target"])[1]
        )
    )

    rlv_info = {}
    rlv_info["result"] = w_x_true

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss, rlv_info


class RNNsearchLRP(NMTModel):
    def __init__(self, params, scope="rnnsearch"):
        super(RNNsearchLRP, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=tf.AUTO_REUSE):
                loss = model_graph(features, features["target"], params)[0]
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return evaluation_fn

    def get_relevance_func(self):
        def relevance_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                loss, rlv = model_graph(features, features["target"],
                                   params)
                return features["source"] , features["target"], rlv, loss
        return relevance_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "rnnsearch"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            unk="<unk>",
            eos="<eos>",
            bos="<eos>",
            append_eos=False,
            # model
            rnn_cell="LegacyGRUCell",
            embedding_size=620,
            hidden_size=1000,
            maxnum=2,
            # regularization
            dropout=0.2,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=True,
            batch_size=128,
            max_length=60,
            clip_grad_norm=5.0,
            #lrp
            stab = 0.05
        )

        return params
