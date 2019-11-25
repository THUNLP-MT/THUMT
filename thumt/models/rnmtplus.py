# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
import thumt.modules as modules


class BiLSTMLayer(modules.Module):

    def __init__(self, input_size, hidden_size, name="layer"):
        super(BiLSTMLayer, self).__init__(name=name)

        with utils.scope(name):
            with utils.scope("forward"):
                self.fwd_cell = modules.LSTMCell(
                    input_size, hidden_size)
            with utils.scope("backward"):
                self.bwd_cell = modules.LSTMCell(
                    input_size, hidden_size)

    def forward(self, x, mask):
        outputs = []
        prev_state = self.fwd_cell.init_state(x.shape[0], x.dtype, x.device)

        for i in range(x.shape[1]):
            inputs = x[:, i]
            mask_t = mask[:, i]
            output, new_state = self.fwd_cell(inputs, prev_state)
            outputs.append(torch.unsqueeze(output, 1))
            new_state = self.fwd_cell.mask_state(new_state, prev_state, mask_t)
            prev_state = new_state

        y_fwd = torch.cat(outputs, 1)

        outputs = []
        prev_state = self.bwd_cell.init_state(x.shape[0], x.dtype, x.device)

        for i in range(x.shape[1] - 1, -1, -1):
            inputs = x[:, i]
            mask_t = mask[:, i]
            output, new_state = self.bwd_cell(inputs, prev_state)
            outputs.append(torch.unsqueeze(output, 1))
            new_state = self.fwd_cell.mask_state(new_state, prev_state, mask_t)
            prev_state = new_state

        y_bwd = torch.cat(outputs[::-1], 1)

        return torch.cat([y_fwd, y_bwd], dim=-1)



class RNMTPlusEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(RNMTPlusEncoder, self).__init__(name=name)

        with utils.scope(name):
            self.layers = nn.ModuleList([
                BiLSTMLayer(params.hidden_size, params.hidden_size,
                            name="layer_%d" % i)
                if i == 0 else
                BiLSTMLayer(2 * params.hidden_size, params.hidden_size,
                            name="layer_%d" % i)
                for i in range(params.num_encoder_layers)])
            self.proj = modules.Affine(2 * params.hidden_size,
                                       params.hidden_size,
                                       name="proj")

        self.dropout = params.residual_dropout

    def forward(self, x, mask):
        for i, layer in enumerate(self.layers):
            y = layer(x, mask)

            if i < 2:
                x = y
            else:
                x = x + nn.functional.dropout(y, self.dropout, self.training)

        return self.proj(x)


class AttentionLSTMLayer(modules.Module):

    def __init__(self, hidden_size, num_heads, dropout,
                 name="layer"):
        super(AttentionLSTMLayer, self).__init__(name=name)

        with utils.scope(name):
            self.cell = modules.LSTMCell(2 * hidden_size, hidden_size)
            self.attention = modules.MultiHeadAdditiveAttention(
                2 * hidden_size, hidden_size, hidden_size, num_heads, dropout)

    def forward(self, x, memory, mask, bias, state=None):
        outputs = []
        contexts = []
        state = state or self.cell.init_state(x.shape[0], x.dtype, x.device)
        cache = self.attention.compute_cache(memory)

        for i in range(x.shape[1]):
            inputs = x[:, i]
            mask_t = mask[:, i]
            query = torch.cat([inputs, state[1]], dim=-1)[:, None, :]
            context = self.attention(query, bias, memory, cache=cache)
            inputs = torch.cat([inputs, torch.squeeze(context, 1)], -1)
            output, new_state = self.cell(inputs, state)
            outputs.append(torch.unsqueeze(output, 1))
            contexts.append(context)
            new_state = self.cell.mask_state(new_state, state, mask_t)
            state = new_state


        output = torch.cat(outputs, 1)
        context = torch.cat(contexts, 1)

        return output, context, state


class LSTMLayer(modules.Module):

    def __init__(self, input_size, hidden_size, name="layer"):
        super(LSTMLayer, self).__init__(name=name)

        with utils.scope(name):
            self.cell = modules.LSTMCell(2 * hidden_size, hidden_size)

    def forward(self, x, context, mask, state=None):
        outputs = []
        state = state or self.cell.init_state(x.shape[0], x.dtype, x.device)

        for i in range(x.shape[1]):
            inputs = x[:, i]
            ctx = context[:, i]
            mask_t = mask[:, i]
            inputs = torch.cat([inputs, ctx], dim=-1)
            output, new_state = self.cell(inputs, state)
            outputs.append(torch.unsqueeze(output, 1))
            new_state = self.cell.mask_state(new_state, state, mask_t)
            state = new_state

        output = torch.cat(outputs, 1)

        return output, state


class RNMTPlusDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(RNMTPlusDecoder, self).__init__(name=name)

        with utils.scope(name):
            self.layers = nn.ModuleList([
                AttentionLSTMLayer(params.hidden_size, params.num_heads,
                                   params.attention_dropout,
                                   name="layer_%d" % i)
                if i == 0 else
                LSTMLayer(2 * params.hidden_size, params.hidden_size,
                          name="layer_%d" % i)
                for i in range(params.num_decoder_layers)])

        self.num_layers = params.num_decoder_layers
        self.dropout = params.residual_dropout

    def forward(self, x, memory, mask, bias, state=None):
        y, ctx, s = self.layers[0](x, memory, mask, bias, state["layer_0"])
        x = y
        state["layer_0"] = s

        for i in range(1, self.num_layers):
            y, s = self.layers[i](x, ctx, mask, state["layer_%d" % i])
            state["layer_%d" % i] = s

            if i < 2:
                x = y
            else:
                x = x + nn.functional.dropout(y, self.dropout, self.training)

        return x


class RNMTPlus(modules.Module):
    # Experimental RNMT+ implementation

    def __init__(self, params, name="rnmtplus"):
        super(RNMTPlus, self).__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoder = RNMTPlusEncoder(params)
            self.decoder = RNMTPlusDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.reset_parameters()

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        if params.shared_source_target_embedding and svoc_size != tvoc_size:
            raise ValueError("Cannot share source and target embedding.")

        if not params.shared_embedding_and_softmax_weights:
            self.softmax_weights = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.softmax_weights, "softmax_weights")

        if not params.shared_source_target_embedding:
            self.source_embedding = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.target_embedding = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.source_embedding, "source_embedding")
            self.add_name(self.target_embedding, "target_embedding")
        else:
            self.weights = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.add_name(self.weights, "weights")

    @property
    def src_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.source_embedding

    @property
    def tgt_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.target_embedding

    @property
    def softmax_embedding(self):
        if not self.params.shared_embedding_and_softmax_weights:
            return self.softmax_weights
        else:
            return self.tgt_embedding

    def reset_parameters(self):
        nn.init.uniform_(self.src_embedding, -0.04, 0.04)
        nn.init.uniform_(self.tgt_embedding, -0.04, 0.04)

        if not self.params.shared_embedding_and_softmax_weights:
            nn.init.uniform_(self.softmax_weights, -0.04, 0.04)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = nn.functional.dropout(inputs, self.dropout, self.training)

        enc_attn_bias = self.masking_bias(src_mask).to(inputs)
        src_mask = src_mask.to(inputs)
        encoder_output = self.encoder(inputs, src_mask)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]
        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        decoder_input = nn.functional.dropout(targets, self.dropout,
                                              self.training)

        tgt_mask = features["target_mask"].to(decoder_input)
        enc_attn_bias = state["enc_attn_bias"]
        encoder_output = state["encoder_output"]

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]

        decoder_output = self.decoder(decoder_input, encoder_output,
                                      tgt_mask, enc_attn_bias,
                                      state["decoder"])

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state

    def forward(self, features, labels):
        mask = features["target_mask"]
        batch_size = features["target"].shape[0]

        state = self.empty_state(batch_size, labels.device)
        state = self.encode(features, state)
        logits, _ = self.decode(features, state, "train")
        loss = self.criterion(logits, labels)
        mask = mask.to(logits)

        # Sentence-level loss
        return torch.mean(torch.sum(loss * mask, 1))

    def empty_state(self, batch_size, device):
        state = {
            "decoder": {
                "layer_%d" % i: (
                    torch.zeros([batch_size, self.hidden_size], device=device),
                    torch.zeros([batch_size, self.hidden_size], device=device)
                ) for i in range(self.num_decoder_layers)
            }
        }

        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def wmt14_ende_1024():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=1024,
            num_heads=4,
            num_encoder_layers=6,
            num_decoder_layers=8,
            attention_dropout=0.3,
            residual_dropout=0.3,
            label_smoothing=0.1,
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            warmup_steps=16000,
            start_decay_step=18750,
            end_decay_step=37500,
            train_steps=100000,
            learning_rate=1e-4,
            learning_rate_schedule="linear_exponential_decay",
            batch_size=128,
            max_length=80,
            fixed_batch_size=True,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-6,
            clip_grad_norm=0.0
        )

        return params

    @staticmethod
    def wmt14_enfr_1024():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=1024,
            num_heads=4,
            num_encoder_layers=6,
            num_decoder_layers=8,
            attention_dropout=0.2,
            residual_dropout=0.2,
            label_smoothing=0.1,
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            warmup_steps=16000,
            start_decay_step=37500,
            end_decay_step=112500,
            train_steps=300000,
            learning_rate=1e-4,
            learning_rate_schedule="linear_exponential_decay",
            batch_size=128,
            max_length=80,
            fixed_batch_size=True,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-6,
            clip_grad_norm=0.0
        )

        return params

    @staticmethod
    def default_params(name=None):
        return RNMTPlus.wmt14_ende_1024()
