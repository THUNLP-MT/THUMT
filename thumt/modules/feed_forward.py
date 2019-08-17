# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size

        self.input_transform = nn.Linear(input_size, hidden_size)
        self.output_transform = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.reset_parameters()

    def forward(self, x):
        h = self.relu(self.input_transform(x))
        return self.output_transform(self.dropout(h))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)
