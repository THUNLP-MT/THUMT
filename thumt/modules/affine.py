# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
from thumt.modules.module import Module


class Affine(Module):

    def __init__(self, in_features, out_features, bias=True, name="affine"):
        super(Affine, self).__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features

        with utils.scope(name):
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.add_name(self.weight, "weight")
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_features))
                self.add_name(self.bias, "bias")
            else:
                self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
