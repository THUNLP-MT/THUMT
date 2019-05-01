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
import thumt.utils.weight_ratio as wr

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


def get_weights(params):
    svocab = params.vocabulary["source"]
    tvocab = params.vocabulary["target"]
    src_vocab_size = len(svocab)
    tgt_vocab_size = len(tvocab)
    vocab_size = tgt_vocab_size

    hidden_size = params.hidden_size
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        if cmp(svocab, tvocab) != 0:
            raise ValueError("Source and target vocabularies are not the same")

        weights = tf.get_variable("weights", [src_vocab_size, hidden_size],
                                  initializer=initializer)
        semb, temb = weights, weights
    else:
        semb = tf.get_variable("source_embedding",
                               [src_vocab_size, hidden_size],
                               initializer=initializer)
        temb = tf.get_variable("target_embedding",
                               [tgt_vocab_size, hidden_size],
                               initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        softmax_weights = temb
    else:
        softmax_weights = tf.get_variable("softmax", [vocab_size, hidden_size],
                                          initializer=initializer)

    return semb, temb, softmax_weights


def layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def residual_fn(x, y, w_x_last, w_x_inp, params, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    batchsize=tf.shape(x)[0]
    len_inp=tf.shape(x)[1]
    len_src = tf.shape(w_x_last)[1]
    dim = tf.shape(x)[2]
    result = {}
    result["output"] = x+y
    x_down = tf.reshape(x, [batchsize, -1])
    y_down = tf.reshape(y, [batchsize, -1])
    z_down = tf.reshape(result["output"], [batchsize, -1])

    w_last_out, w_inp_out = wr.weight_ratio_weighted_sum([x_down,y_down],
                                                         [1.,1.],
                                                         z_down,
                                                         stab=params.stab,
                                                         flatten=True)
    # bs, len*d
    w_last_out = tf.reshape(w_last_out, [batchsize, 1, len_inp, dim])
    w_inp_out = tf.reshape(w_inp_out, [batchsize, 1, len_inp, dim])
    w_x_out = w_x_last * w_last_out + w_x_inp * w_inp_out
    result["weight_ratio"] = w_x_out
    return result


def ffn_layer(inputs, w_x_inp, hidden_size, output_size, params,
              keep_prob=None, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden_linear = lrp.linear_v2n(inputs, hidden_size, True,
                                           [w_x_inp], params, True)
            hidden = hidden_linear["output"]
            w_x_hid = hidden_linear["weight_ratios"][0]
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output_linear = lrp.linear_v2n(hidden, output_size, True,
                                           [w_x_hid], params, True)
            output = output_linear["output"]
            w_x_outp = output_linear["weight_ratios"][0]

        return {"output": output, "weight_ratios": w_x_outp}


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        len_src = tf.shape(inputs)[1]
        batchsize = tf.shape(inputs)[0]
        dim = tf.shape(inputs)[2]
        x = inputs
        w_x_last = lrp.create_diagonal_v2n(batchsize, len_src, dim)
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y_self = lrp.multihead_attention_v2n(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        w_x_last,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params,
                        1.0 - params.attention_dropout
                    )
                    y = y_self["outputs"]
                    w_x_self = y_self["weight_ratio"]

                    x_res = residual_fn(x, y, w_x_last, w_x_self, params,
                                        1.0 - params.residual_dropout)
                    x = x_res["output"]
                    w_x_selfres = x_res["weight_ratio"]

                    x_norm = lrp.layer_process(x, params.layer_postprocess,
                                               w_x_selfres, params)
                    x = x_norm["outputs"]
                    w_x_selfres = x_norm["weight_ratios"]

                with tf.variable_scope("feed_forward"):
                    y_ffn = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        w_x_selfres,
                        params.filter_size,
                        params.hidden_size,
                        params,
                        1.0 - params.relu_dropout,
                    )
                    y = y_ffn["output"]
                    w_x_ffn = y_ffn["weight_ratios"]

                    x_res = residual_fn(x, y, w_x_selfres, w_x_ffn, params,
                                        1.0 - params.residual_dropout)
                    x = x_res["output"]
                    w_x_ffnres = x_res["weight_ratio"]

                    x_norm = lrp.layer_process(x, params.layer_postprocess,
                                               w_x_ffnres, params)
                    x = x_norm["outputs"]
                    w_x_ffnres = x_norm["weight_ratios"]

                    w_x_last = w_x_ffnres

        outputs = layer_process(x, params.layer_preprocess)
        return {"outputs": outputs, "weight_ratios": w_x_last}


def transformer_decoder(inputs, memory, bias, mem_bias, w_x_enc, params,
                        dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        len_src = tf.shape(memory)[1]
        len_trg = tf.shape(inputs)[1]
        batchsize = tf.shape(inputs)[0]
        dim = tf.shape(inputs)[2]
        x = inputs
        w_x_last = tf.zeros([batchsize, len_src, len_trg, dim],
                            dtype=tf.float32)
        for layer in range(params.num_decoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y_self = lrp.multihead_attention_v2n(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        w_x_last,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params,
                        1.0 - params.attention_dropout
                    )
                    y = y_self["outputs"]
                    w_x_self = y_self["weight_ratio"]

                    x_res = residual_fn(x, y,w_x_last,w_x_self,params,
                                        1.0 - params.residual_dropout)
                    x = x_res["output"]
                    w_x_selfres = x_res["weight_ratio"]

                    x_norm = lrp.layer_process(x, params.layer_postprocess,
                                               w_x_selfres, params)
                    x = x_norm["outputs"]
                    w_x_selfres = x_norm["weight_ratios"]

                with tf.variable_scope("encdec_attention"):
                    y_encdec = lrp.multihead_attention_v2n(
                        layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        w_x_enc,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params,
                        1.0 - params.attention_dropout
                    )
                    y = y_encdec["outputs"]
                    w_x_encdec = y_encdec["weight_ratio"]

                    x_res = residual_fn(x, y, w_x_selfres, w_x_encdec, params,
                                        1.0 - params.residual_dropout)
                    x = x_res["output"]
                    w_x_encdecres = x_res["weight_ratio"]

                    x_norm = lrp.layer_process(x, params.layer_postprocess,
                                               w_x_encdecres, params)
                    x = x_norm["outputs"]
                    w_x_encdecres = x_norm["weight_ratios"]

                with tf.variable_scope("feed_forward"):
                    y_ffn = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        w_x_encdecres,
                        params.filter_size,
                        params.hidden_size,
                        params,
                        1.0 - params.relu_dropout,
                    )
                    y = y_ffn["output"]
                    w_x_ffn = y_ffn["weight_ratios"]

                    x_res = residual_fn(x, y, w_x_encdecres, w_x_ffn, params,
                                        1.0 - params.residual_dropout)
                    x = x_res["output"]
                    w_x_ffnres = x_res["weight_ratio"]

                    x_norm = lrp.layer_process(x, params.layer_postprocess,
                                               w_x_ffnres,params)
                    x = x_norm["outputs"]
                    w_x_ffnres = x_norm["weight_ratios"]

                    w_x_last = w_x_ffnres

        outputs = layer_process(x, params.layer_preprocess)

        return  {"outputs": outputs, "weight_ratios": w_x_last}


def model_graph(features, labels, mode, params):
    hidden_size = params.hidden_size

    src_seq = features["source"]
    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    src_embedding, tgt_embedding, weights = get_weights(params)
    bias = tf.get_variable("bias", [hidden_size])

    # id => embedding
    # src_seq: [batch, max_src_length]
    # tgt_seq: [batch, max_tgt_length]
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder & decoder input
    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")

    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_out = transformer_encoder(encoder_input, enc_attn_bias, params)
    encoder_output = encoder_out["outputs"]
    w_x_enc = encoder_out["weight_ratios"]
    decoder_out = transformer_decoder(decoder_input, encoder_output,
                                      dec_attn_bias, enc_attn_bias,
                                      w_x_enc, params)
    decoder_output = decoder_out["outputs"]
    w_x_dec = decoder_out["weight_ratios"]

    weights_true = tf.gather(weights, labels)
    logits_elewise_true = decoder_output * weights_true
    logits_true = tf.reduce_sum(logits_elewise_true, -1)
    logits_stab = lrp.stabilize(tf.expand_dims(logits_true, -1), params.stab)
    wr_logit_decoder = logits_elewise_true / logits_stab
    w_x_true = w_x_dec * tf.expand_dims(wr_logit_decoder, 1)
    w_x_true = tf.reduce_sum(w_x_true, -1)
    # inference mode, take the last position
    if mode == "infer":
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)

        return logits

    # [batch, length, channel] => [batch * length, vocab_size]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)

    # label smoothing
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))
    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    rlv_info = {}
    R_x_true = tf.transpose(w_x_true, [0, 2, 1])
    rlv_info["result"] = normalize(R_x_true, True)

    return loss, rlv_info


class TransformerLRP(NMTModel):
    def __init__(self, params, scope="transformer"):
        super(TransformerLRP, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=tf.AUTO_REUSE):
                loss = model_graph(features, features["target"],
                                   "train", params)[0]
                return loss

        return training_fn

    def get_relevance_func(self):
        def relevance_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.residual_dropout = 0.0
            params.attention_dropout = 0.0
            params.relu_dropout = 0.0
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
                loss, rlv = model_graph(features, features["target"],
                                   "train", params)
                return features["source"] , features["target"], rlv, loss
        return relevance_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.residual_dropout = 0.0
            params.attention_dropout = 0.0
            params.relu_dropout = 0.0
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, "infer", params)

            return logits

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.residual_dropout = 0.0
            params.attention_dropout = 0.0
            params.relu_dropout = 0.0
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, "infer", params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="noam",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            # lrp
            stab=0.05,
        )

        return params
