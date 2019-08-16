# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.distributed as dist

from thumt.optimizers.optimizer import Optimizer
from thumt.optimizers.schedules import LearningRateSchedule


class AdamOptimizer(Optimizer):

    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-7, name="Adam", **kwargs):
        super(AdamOptimizer, self).__init__(name, **kwargs)
        self._hyper["learning_rate"] = learning_rate
        self._hyper["beta_1"] = beta_1
        self._hyper["beta_2"] = beta_2
        self._hyper["epsilon"] = epsilon

    def apply_gradients(self, grads_and_vars):
        self._iterations += 1
        lr = self._hyper["learning_rate"]
        beta_1 = self._hyper["beta_1"]
        beta_2 = self._hyper["beta_2"]
        epsilon = self._hyper["epsilon"]

        for grad, var in grads_and_vars:
            if grad is None:
                continue

            if dist.get_world_size() > 1:
                dist.all_reduce(grad.data, op=dist.reduce_op.SUM)
                grad.data /= dist.get_world_size()

            if self._slots.get(var, None) is None:
                self._slots[var] = {}
                self._slots[var]["m"] = torch.zeros_like(var.data)
                self._slots[var]["v"] = torch.zeros_like(var.data)

            m, v = self._slots[var]["m"], self._slots[var]["v"]

            bias_corr_1 = 1 - beta_1 ** self._iterations
            bias_corr_2 = 1 - beta_2 ** self._iterations

            m.mul_(beta_1).add_(1 - beta_1, grad)
            v.mul_(beta_2).addcmul_(1 - beta_2, grad, grad)
            denom = (v.sqrt() / math.sqrt(bias_corr_2)).add_(epsilon)

            if isinstance(lr, LearningRateSchedule):
                lr = lr(self._iterations)

            step_size = lr / bias_corr_1
            var.data.addcdiv_(-step_size, m, denom)
