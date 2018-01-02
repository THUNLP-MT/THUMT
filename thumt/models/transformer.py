# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers


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


def residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
              dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

        outputs = layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, dtype=None,
                        scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        for layer in range(params.num_decoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

        outputs = layer_process(x, params.layer_preprocess)

        return outputs


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

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)
    decoder_output = transformer_decoder(decoder_input, encoder_output,
                                         dec_attn_bias, enc_attn_bias, params)

    # inference mode, take the last position
    if mode == "infer":
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)

        return logits

    # [batch, length, channel] => [batch * length, vocab_size]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))
    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


class Transformer(interface.NMTModel):
    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=tf.AUTO_REUSE):
                loss = model_graph(features, features["target"],
                                   "train", params)
                return loss

        return training_fn

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
            clip_grad_norm=0.0
        )

        return params
