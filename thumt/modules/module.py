# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import thumt.utils as utils


class Module(nn.Module):

    def __init__(self, name=""):
        super(Module, self).__init__()
        scope = utils.get_scope()
        self._name = scope + "/" + name if scope else name

    def add_name(self, tensor, name):
        tensor.tensor_name = utils.unique_name(name)

    @property
    def name(self):
        return self._name
