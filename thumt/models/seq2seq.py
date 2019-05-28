# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils as utils

from thumt.models.model import NMTModel


def model_graph(features, mode, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])
    dtype = tf.get_variable_scope().dtype

    src_seq = features["source"]
    tgt_seq = features["target"]

    if params.reverse_source:
        src_seq = tf.reverse_sequence(src_seq, seq_dim=1,
                                      seq_lengths=features["source_length"])

    with tf.device("/cpu:0"):
        with tf.variable_scope("source_embedding"):
            src_emb = tf.get_variable("embedding",
                                      [src_vocab_size, params.embedding_size])
            src_bias = tf.get_variable("bias", [params.embedding_size])
            src_inputs = tf.nn.embedding_lookup(src_emb, src_seq)

        with tf.variable_scope("target_embedding"):
            tgt_emb = tf.get_variable("embedding",
                                      [tgt_vocab_size, params.embedding_size])
            tgt_bias = tf.get_variable("bias", [params.embedding_size])
            tgt_inputs = tf.nn.embedding_lookup(tgt_emb, tgt_seq)

    src_inputs = tf.nn.bias_add(src_inputs, src_bias)
    tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)

    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
        tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

    cell_enc = []
    cell_dec = []

    for _ in range(params.num_hidden_layers):
        if params.rnn_cell == "LSTMCell":
            cell_e = tf.nn.rnn_cell.BasicLSTMCell(params.hidden_size)
            cell_d = tf.nn.rnn_cell.BasicLSTMCell(params.hidden_size)
        elif params.rnn_cell == "GRUCell":
            cell_e = tf.nn.rnn_cell.GRUCell(params.hidden_size)
            cell_d = tf.nn.rnn_cell.GRUCell(params.hidden_size)
        else:
            raise ValueError("%s not supported" % params.rnn_cell)

        cell_e = tf.nn.rnn_cell.DropoutWrapper(
            cell_e,
            output_keep_prob=1.0 - params.dropout,
            variational_recurrent=params.use_variational_dropout,
            input_size=params.embedding_size,
            dtype=dtype
        )
        cell_d = tf.nn.rnn_cell.DropoutWrapper(
            cell_d,
            output_keep_prob=1.0 - params.dropout,
            variational_recurrent=params.use_variational_dropout,
            input_size=params.embedding_size,
            dtype=dtype
        )

        if params.use_residual:
            cell_e = tf.nn.rnn_cell.ResidualWrapper(cell_e)
            cell_d = tf.nn.rnn_cell.ResidualWrapper(cell_d)

        cell_enc.append(cell_e)
        cell_dec.append(cell_d)

    cell_enc = tf.nn.rnn_cell.MultiRNNCell(cell_enc)
    cell_dec = tf.nn.rnn_cell.MultiRNNCell(cell_dec)

    with tf.variable_scope("encoder"):
        _, final_state = tf.nn.dynamic_rnn(cell_enc, src_inputs,
                                           features["source_length"],
                                           dtype=dtype)
    # Shift left
    shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
    shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]

    with tf.variable_scope("decoder"):
        outputs, _ = tf.nn.dynamic_rnn(cell_dec, shifted_tgt_inputs,
                                       features["target_length"],
                                       initial_state=final_state)

    if params.dropout:
        outputs = tf.nn.dropout(outputs, 1.0 - params.dropout)

    if mode == "infer":
        # Prediction
        logits = layers.nn.linear(outputs[:, -1, :], tgt_vocab_size, True,
                                  scope="softmax")

        return tf.nn.log_softmax(logits)

    # Prediction
    logits = layers.nn.linear(outputs, tgt_vocab_size, True, scope="softmax")
    logits = tf.reshape(logits, [-1, tgt_vocab_size])
    labels = features["target"]

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

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


class Seq2Seq(NMTModel):

    def __init__(self, params, scope="seq2seq"):
        super(Seq2Seq, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters

            custom_getter = utils.custom_getter if dtype else None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, "train", params)
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
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

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
                logits = model_graph(features, "infer", params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "seq2seq"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            # model
            rnn_cell="LSTMCell",
            embedding_size=1000,
            hidden_size=1000,
            num_hidden_layers=4,
            # regularization
            dropout=0.2,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=True,
            batch_size=128,
            max_length=80,
            reverse_source=True,
            use_residual=True,
            clip_grad_norm=5.0
        )

        return params
