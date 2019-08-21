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
        self._iterations = 0

    def detach_gradients(self, gradients):
        for grad in gradients:
            if grad is not None:
                grad.detach_()

    def scale_gradients(self, gradients, scale):
        for grad in gradients:
            if grad is not None:
                grad.mul_(scale)

    def sync_gradients(self, gradients, compress=True):
        for grad in gradients:
            if grad is None:
                continue

            if compress and grad.dtype != torch.float16:
                grad_fp16 = grad.half()
                dist.all_reduce(grad_fp16)
                grad.copy_(grad_fp16)
            else:
                dist.all_reduce(grad)

    def zero_gradients(self, gradients):
        for grad in gradients:
            if grad is not None:
                grad.zero_()

    def compute_gradients(self, loss, var_list, aggregate=False):
        var_list = list(var_list)
        grads = [v.grad if v is not None else None for v in var_list]

        self.detach_gradients(grads)

        if not aggregate:
            self.zero_gradients(grads)

        loss.backward()
        return [v.grad if v is not None else None for v in var_list]

    def apply_gradients(self, grads_and_vars):
        raise NotImplementedError("Not implemented")

    @property
    def iterations(self):
        return self._iterations

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

            # Convert if grad is not FP32
            grad = grad.data.float()

            if self._slots.get(var, None) is None:
                self._slots[var] = {}
                self._slots[var]["m"] = torch.zeros_like(var.data,
                                                         dtype=torch.float32)
                self._slots[var]["v"] = torch.zeros_like(var.data,
                                                         dtype=torch.float32)

            m, v = self._slots[var]["m"], self._slots[var]["v"]

            bias_corr_1 = 1 - beta_1 ** self._iterations
            bias_corr_2 = 1 - beta_2 ** self._iterations

            m.mul_(beta_1).add_(1 - beta_1, grad)
            v.mul_(beta_2).addcmul_(1 - beta_2, grad, grad)
            denom = (v.sqrt() / math.sqrt(bias_corr_2)).add_(epsilon)

            if isinstance(lr, LearningRateSchedule):
                lr = lr(self._iterations)

            step_size = lr / bias_corr_1

            if var.dtype == torch.float32:
                var.data.addcdiv_(-step_size, m, denom)
            else:
                fp32_var = var.data.float()
                fp32_var.addcdiv_(-step_size, m, denom)
                var.data.copy_(fp32_var)


class LossScalingOptimizer(Optimizer):

    def __init__(self, optimizer, scale=2.0**7, increment_period=2000,
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

    def compute_gradients(self, loss, var_list, aggregate=False):
        var_list = list(var_list)
        grads = [v.grad if v is not None else None for v in var_list]

        self.detach_gradients(grads)

        if not aggregate:
            self.zero_gradients(grads)

        loss = loss * self._scale
        loss.backward()

        return [v.grad if v is not None else None for v in var_list]

    def apply_gradients(self, grads_and_vars):
        grads, var_list = list(zip(*grads_and_vars))
        new_grads = []

        for grad in grads:
            if grad is None:
                new_grads.append(None)
                continue

            norm = grad.data.norm()

            if not torch.isfinite(norm):
                self._update_if_not_finite_grads()
                return
            else:
                # Rescale gradients
                new_grads.append(grad.data.float().mul_(1.0 / self._scale))

        self._update_if_finite_grads()
        self._optimizer.apply_gradients(zip(new_grads, var_list))


class MultiStepOptimizer(Optimizer):

    def __init__(self, optimizer, n=1, compress=True,
                 name="MultiStepOptimizer", **kwargs):
        super(MultiStepOptimizer, self).__init__(name, **kwargs)
        self._n = n
        self._iterations = 0
        self._optimizer = optimizer
        self._compress = compress

    def compute_gradients(self, loss, var_list, aggregate=False):
        if self._iterations % self._n == 0:
            return self._optimizer.compute_gradients(loss, var_list, aggregate)
        else:
            return self._optimizer.compute_gradients(loss, var_list, True)

    def apply_gradients(self, grads_and_vars):
        size = dist.get_world_size()
        grads, var_list = list(zip(*grads_and_vars))
        self._iterations += 1

        if self._n == 1:
            if size > 1:
                self.sync_gradients(grads, compress=self._compress)
                self.scale_gradients(grads, 1.0 / size)

            self._optimizer.apply_gradients(zip(grads, var_list))
        else:
            if self._iterations % self._n != 0:
                return

            if size > 1:
                self.sync_gradients(grads, compress=self._compress)

            self.scale_gradients(grads, 1.0 / (self._n * size))
            self._optimizer.apply_gradients(zip(grads, var_list))
