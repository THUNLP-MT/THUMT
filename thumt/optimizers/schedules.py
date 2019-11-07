# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import thumt.utils as utils
import thumt.utils.summary as summary


class LearningRateSchedule(object):

    def __call__(self, step):
        raise NotImplementedError("Not implemented.")

    def get_config(self):
      raise NotImplementedError("Not implemented.")

    @classmethod
    def from_config(cls, config):
      return cls(**config)



class LinearWarmupRsqrtDecay(LearningRateSchedule):

    def __init__(self, learning_rate, warmup_steps, initial_learning_rate=0.0,
                 summary=True):
        super(LinearWarmupRsqrtDecay, self).__init__()

        if not initial_learning_rate:
            initial_learning_rate = initial_learning_rate / warmup_steps

        self._initial_learning_rate = initial_learning_rate
        self._maximum_learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._summary = summary

    def __call__(self, step):
        if step <= self._warmup_steps:
            lr_step = self._maximum_learning_rate - self._initial_learning_rate
            lr_step /= self._warmup_steps
            lr = self._initial_learning_rate + lr_step * step
        else:
            step = step / self._warmup_steps
            lr = self._maximum_learning_rate * (step ** -0.5)

        if self._summary:
            summary.scalar("learning_rate", lr, utils.get_global_step())

        return lr

    def get_config(self):
        return {
            "learning_rate": self._maximum_learning_rate,
            "initial_learning_rate": self._initial_learning_rate,
            "warmup_steps": self._warmup_steps
        }
