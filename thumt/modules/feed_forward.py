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


class FeedForward(Module):

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0,
                 name="feed_forward"):
        super(FeedForward, self).__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.dropout = dropout

        with utils.scope(name):
            self.input_transform = Affine(input_size, hidden_size,
                                          name="input_transform")
            self.output_transform = Affine(hidden_size, self.output_size,
                                           name="output_transform")

        self.reset_parameters()

    def forward(self, x):
        h = nn.functional.relu(self.input_transform(x))
        h = nn.functional.dropout(h, self.dropout, self.training)
        return self.output_transform(h)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)
