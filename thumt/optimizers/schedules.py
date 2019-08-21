# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class LearningRateSchedule(object):

    def __call__(self, step):
        raise NotImplementedError("Not implemented.")

    def get_config(self):
      raise NotImplementedError("Not implemented.")

    @classmethod
    def from_config(cls, config):
      return cls(**config)



class LinearWarmupRsqrtDecay(LearningRateSchedule):

    def __init__(self, learning_rate, warmup_steps, initial_learning_rate=0.0):
        super(LinearWarmupRsqrtDecay, self).__init__()

        if not initial_learning_rate:
            initial_learning_rate = initial_learning_rate / warmup_steps

        self.initial_learning_rate = initial_learning_rate
        self.maximum_learning_rate = learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step <= self.warmup_steps:
            lr_step = self.maximum_learning_rate - self.initial_learning_rate
            lr_step /= self.warmup_steps
            return self.initial_learning_rate + lr_step * step
        else:
            step = step / self.warmup_steps
            return self.maximum_learning_rate * (step ** -0.5)

    def get_config(self):
        return {
            "learning_rate": self.maximum_learning_rate,
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps
        }
