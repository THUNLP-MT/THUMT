# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.distributed as dist


class Optimizer(object):

    def __init__(self, name, **kwargs):
        self._name = name
        self._hyper = {}
        self._slots = {}
        self._weights = None
        self._iterations = 0

    def _zero_gradients(self, var_list):
        for v in var_list:
            if v is not None and v.grad is not None:
                v.grad.detach_()
                v.grad.zero_()

    def _reduce_gradients(self, gradients):
        for grad in gradients:
            dist.all_reduce(grad.data, op=dist.reduce_op.SUM)
            grad.data /= dist.get_world_size()

    def compute_gradients(self, loss, var_list):
        var_list = list(var_list)
        self._zero_gradients(var_list)
        loss.backward()
        return [v.grad if v is not None else None for v in var_list]

    def apply_gradients(self, grads_and_vars):
        raise NotImplementedError("Not implemented")

    @property
    def iterations(self):
        return self._iterations

    @property
    def weights(self):
        return self._weights
