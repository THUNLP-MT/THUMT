# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch


class PositionalEmbedding(torch.nn.Module):

    def __init__(self):
        super(PositionalEmbedding, self).__init__()

    def forward(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("The rank of input must be 3.")

        length = inputs.shape[1]
        channels = inputs.shape[2]
        half_dim = channels // 2

        positions = torch.arange(length, dtype=inputs.dtype,
                                 device=inputs.device)
        dimensions = torch.arange(half_dim, dtype=inputs.dtype,
                                  device=inputs.device)

        scale = math.log(10000.0) / float(half_dim - 1)
        dimensions.mul_(-scale).exp_()

        scaled_time = positions.unsqueeze(1) * dimensions.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)

        if channels % 2 == 1:
            pad = torch.zeros([signal.shape[0], 1], dtype=inputs.dtype,
                              device=inputs.device)
            signal = torch.cat([signal, pad], axis=1)

        return inputs + torch.reshape(signal, [1, -1, channels]).to(inputs)
