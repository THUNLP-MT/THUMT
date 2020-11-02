# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import thumt.utils as utils

from thumt.modules.module import Module
from thumt.modules.affine import Affine
from thumt.modules.layer_norm import LayerNorm


class GRUCell(Module):

    def __init__(self, input_size, output_size, normalization=False,
                 name="gru"):
        super(GRUCell, self).__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size

        with utils.scope(name):
            self.reset_gate = Affine(input_size + output_size, output_size,
                                     bias=False, name="reset_gate")
            self.update_gate = Affine(input_size + output_size, output_size,
                                      bias=False, name="update_gate")
            self.transform = Affine(input_size + output_size, output_size,
                                    name="transform")

    def forward(self, x, h):
        r = torch.sigmoid(self.reset_gate(torch.cat([x, h], -1)))
        u = torch.sigmoid(self.update_gate(torch.cat([x, h], -1)))
        c = self.transform(torch.cat([x, r * h], -1))

        new_h = (1.0 - u) * h + u * torch.tanh(h)

        return new_h, new_h

    def init_state(self, batch_size, dtype, device):
        h = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        return h

    def mask_state(self, h, prev_h, mask):
        mask = mask[:, None]
        new_h = mask * h + (1.0 - mask) * prev_h
        return new_h

    def reset_parameters(self, initializer="uniform"):
        if initializer == "uniform_scaling":
            nn.init.xavier_uniform_(self.gates.weight)
            nn.init.constant_(self.gates.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.gates.weight, -0.08, 0.08)
            nn.init.uniform_(self.gates.bias, -0.08, 0.08)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class LSTMCell(Module):

    def __init__(self, input_size, output_size, normalization=False,
                 activation=None, name="lstm"):
        super(LSTMCell, self).__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        with utils.scope(name):
            self.gates = Affine(input_size + output_size, 4 * output_size,
                                name="gates")
            if normalization:
                self.layer_norm = LayerNorm([4, output_size])
            else:
                self.layer_norm = None

        self.reset_parameters()

    def forward(self, x, state):
        c, h = state

        gates = self.gates(torch.cat([x, h], 1))

        if self.layer_norm is not None:
            combined = self.layer_norm(
                torch.reshape(gates, [-1, 4, self.output_size]))
        else:
            combined = torch.reshape(gates, [-1, 4, self.output_size])

        i, j, f, o = torch.unbind(combined, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)

        new_c = f * c + i * torch.tanh(j)

        if self.activation is None:
            # Do not use tanh activation
            new_h = o * new_c
        else:
            new_h = o * self.activation(new_c)

        return new_h, (new_c, new_h)

    def init_state(self, batch_size, dtype, device):
        c = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        h = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        return c, h

    def mask_state(self, state, prev_state, mask):
        c, h = state
        prev_c, prev_h = prev_state
        mask = mask[:, None]
        new_c = mask * c + (1.0 - mask) * prev_c
        new_h = mask * h + (1.0 - mask) * prev_h
        return new_c, new_h

    def reset_parameters(self, initializer="uniform"):
        if initializer == "uniform_scaling":
            nn.init.xavier_uniform_(self.gates.weight)
            nn.init.constant_(self.gates.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.gates.weight, -0.04, 0.04)
            nn.init.uniform_(self.gates.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)
