# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
from thumt.modules import MultiHeadAttention


def add_timing_signal(x):
    dtype = x.dtype
    length = x.shape[1]
    channels = x.shape[2]
    half_dim = channels // 2
    positions = torch.arange(length, dtype=dtype, device=x.device)
    dimensions = torch.arange(half_dim, dtype=dtype, device=x.device)
    scale = math.log(10000.0) / float(half_dim - 1)

    dimensions.mul_(-scale).exp_()
    scaled_time = positions.unsqueeze(1) * dimensions.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    if channels % 2 == 1:
        pad = torch.zeros([signal.shape(0), 1], dtype=dtype, device=x.device)
        signal = torch.cat([signal, pad], axis=1)

    signal = torch.reshape(signal, [1, -1, channels])
    return x + signal


class AttentionSubLayer(nn.Module):

    def __init__(self, params):
        super(AttentionSubLayer, self).__init__()
        self.attention = MultiHeadAttention(params.hidden_size,
                                            params.num_heads,
                                            params.attention_dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_size)
        self.dropout_rate = params.residual_dropout

    def forward(self, x, bias, memory=None, state=None):
        y = self.attention(x, bias, memory, state)

        if self.dropout_rate > 0.0:
            y = torch.dropout(y, p=self.dropout_rate, train=True)

        return self.layer_norm(x + y)


class FFNSubLayer(nn.Module):

    def __init__(self, params, dtype=None):
        super(FFNSubLayer, self).__init__()
        hidden_size = params.hidden_size
        filter_size = params.filter_size
        self.input_transform = nn.Linear(hidden_size, filter_size)
        self.output_transform = nn.Linear(filter_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_rate = params.residual_dropout
        self.relu_droput_rate = params.relu_dropout
        self.init_paraemeters()

    def forward(self, x):
        y = self.input_transform(x)

        if self.relu_droput_rate > 0.0:
            y = torch.dropout(y, p=self.dropout_rate, train=True)

        y = self.output_transform(torch.relu(y))

        if self.dropout_rate > 0.0:
            y = torch.dropout(y, p=self.dropout_rate, train=True)

        return self.layer_norm(x + y)

    def init_paraemeters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, params):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = AttentionSubLayer(params)
        self.feed_forward = FFNSubLayer(params)

    def forward(self, x, bias):
        x = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, params):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = AttentionSubLayer(params)
        self.encdec_attention = AttentionSubLayer(params)
        self.feed_forward = FFNSubLayer(params)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        x = self.self_attention(x, attn_bias, state=state)
        x = self.encdec_attention(x, encdec_bias, memory, state=state)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, params):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(params)
            for i in range(params.num_encoder_layers)
        ])

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, params):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(params)
            for i in range(params.num_encoder_layers)
        ])

    def forward(self, x, attn_bias, encdec_bias, memory, state=None):
        for i, layer in enumerate(self.layers):
            if state is not None:
                x = layer(x, attn_bias, encdec_bias, memory,
                          state["decoder"]["layer_%d" % i])
            else:
                x = layer(x, attn_bias, encdec_bias, memory, None)
        return x


class Transformer(nn.Module):

    def __init__(self, params):
        super(Transformer, self).__init__()
        self.build_embedding(params)
        self.encoder = TransformerEncoder(params)
        self.decoder = TransformerDecoder(params)
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.label_smoothing = params.label_smoothing
        self.mode = "train"

    def build_embedding(self, params):
        src_vocab_size = len(params.vocabulary["source"])
        tgt_vocab_size = len(params.vocabulary["target"])

        self.source_embedding = torch.nn.Parameter(
            torch.empty([src_vocab_size, params.hidden_size])
        )

        if params.shared_source_target_embedding:
            self.target_embedding = self.source_embedding
        else:
            self.target_embedding = torch.nn.Parameter(
                torch.empty([tgt_vocab_size, params.hidden_size]))

        if params.shared_embedding_and_softmax_weights:
            self.softmax_weights = self.target_embedding
        else:
            self.softmax_weights = torch.nn.Parameter(
                torch.empty([tgt_vocab_size, params.hidden_size]))

        self.bias = torch.nn.Parameter(torch.zeros([params.hidden_size]))

        nn.init.normal_(self.source_embedding, mean=0,
                        std=params.hidden_size ** -0.5)
        nn.init.normal_(self.target_embedding, mean=0,
                        std=params.hidden_size ** -0.5)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = torch.ne(src_seq, 0).float()
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.source_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs += self.bias
        inputs = add_timing_signal(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state):
        tgt_seq = features["target"]
        tgt_mask = torch.ne(tgt_seq, 0).float()

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = torch.nn.functional.embedding(tgt_seq, self.target_embedding)
        targets = targets * (self.hidden_size ** 0.5)
        targets = targets[:, 1:, :]
        decoder_input = torch.cat(
            [torch.zeros([targets.shape[0], 1, targets.shape[-1]],
                         dtype=targets.dtype, device=targets.device), targets],
            dim=1)
        decoder_input = add_timing_signal(decoder_input)
        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets.device)

        if self.mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, state)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_weights, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state

    def forward(self, features):
        labels = torch.reshape(features["labels"], [-1, 1])
        state = self.empty_state(features["target"].shape[0],
                                 labels.device)
        state = self.encode(features, state)
        logits, _ = self.decode(features, state)

        return logits

    def empty_state(self, batch_size, device):
        state = {
            "decoder": {
                "layer_%d" % i: {
                    "k": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device)
                    ,
                } for i in range(self.num_decoder_layers)
            }
        }

        return state

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def infer(self):
        self.mode = "infer"

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def default_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
        )

        return params
