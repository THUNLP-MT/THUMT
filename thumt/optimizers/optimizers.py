# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.distributed as dist

from thumt.optimizers.schedules import LearningRateSchedule


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

    def state_dict(self):
        raise NotImplementedError("Not implemented")

    def load_state_dict(self):
        raise NotImplementedError("Not implemented")


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


class MultiStepOptimizer(Optimizer):

    def __init__(self, optimizer, n=1, name="MultiStepOptimizer", **kwargs):
        super(MultiStepOptimizer, self).__init__(name, **kwargs)
        self._n = n
        self._iterations = 0
        self._optimizer = optimizer

    def compute_gradients(self, loss, var_list):
        return self._optimizer.compute_gradients(loss, var_list)

    def apply_gradients(self, grads_and_vars):
        self._iterations += 1

        if self._iterations % self._n == 0:
            grads, var_list = list(zip(*grads_and_vars))

            # Average gradient
            for grad in grads:
                grad.data.div_(self._n)

            self._optimizer.apply_gradients(zip(grads, var_list))


class LossScalingOptimizer(Optimizer):

    def __init__(self, optimizer, scale=2.0**15, increment_period=2000,
                 multiplier=2.0, name="LossScalingOptimizer", **kwargs):
        super(LossScalingOptimizer, self).__init__(name, **kwargs)
        self._optimizer = optimizer
        self._scale = scale
        self._skip_update = False
        self._increment_preiod = increment_period
        self._multiplier = multiplier
        self._num_good_steps = 0

    def _update_if_finite_grads(self):
        if self._num_good_steps + 1 > self._increment_preiod:
            self._scale *= self._multiplier
            self._num_good_steps = 0
        else:
            self._num_good_steps += 1

    def _update_if_not_finite_grads(self):
        self._scale = max(self._scale / self._multiplier, 1)

    def compute_gradients(self, loss, var_list):
        var_list = list(var_list)
        self._zero_gradients(var_list)

        self._skip_update = True

        if not torch.isfinite(loss):
            self._skip_update = True
            self._update_if_finite_grads()
        else:
            self._skip_update = False
            loss = loss * self._scale
            loss.backward()

            for v in var_list:
                if v is None:
                    continue

                norm = v.grad.norm()

                if not torch.isfinite(norm):
                    self._skip_update = True
                    self._update_if_finite_grads()
                    break

            self._update_if_finite_grads()

        return [v.grad if v is not None else None for v in var_list]

    def apply_gradients(self, grads_and_vars):
        if self._skip_update:
            return
        else:
            grads = []
            var_list = []

            for grad, var in grads_and_vars:
                if grad is None:
                    continue

                if self._slots.get(var, None) is None:
                    self._slots[var] = torch.zeros_like(var.data,
                                                        dtype=torch.float32)

                v = self._slots[var]
                v.copy_(grad.data)

                grads.append(v)
                var_list.append(var)

            self._optimizer.apply_gradients(self, zip(grads, var_list))

