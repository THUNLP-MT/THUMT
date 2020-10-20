# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

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

        if initial_learning_rate <= 0:
            if warmup_steps > 0:
                initial_learning_rate = learning_rate / warmup_steps
            else:
                initial_learning_rate = 0.0
        elif initial_learning_rate >= learning_rate:
            raise ValueError("The maximum learning rate: %f must be "
                             "higher than the initial learning rate:"
                             " %f" % (learning_rate, initial_learning_rate))

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
            lr = self._maximum_learning_rate

            if self._warmup_steps != 0:
                # approximately hidden_size ** -0.5
                lr = lr * self._warmup_steps ** 0.5

            lr = lr * (step ** -0.5)

        if self._summary:
            summary.scalar("learning_rate", lr, utils.get_global_step())

        return lr

    def get_config(self):
        return {
            "learning_rate": self._maximum_learning_rate,
            "initial_learning_rate": self._initial_learning_rate,
            "warmup_steps": self._warmup_steps
        }


class PiecewiseConstantDecay(LearningRateSchedule):

    def __init__(self, boundaries, values, summary=True):
        super(PiecewiseConstantDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError("The length of boundaries should be 1"
                             " less than the length of values")

        self._boundaries = boundaries
        self._values = values
        self._summary = summary

    def __call__(self, step):
        boundaries = self._boundaries
        values = self._values
        learning_rate = values[0]

        if step <= boundaries[0]:
            learning_rate = values[0]
        elif step > boundaries[-1]:
            learning_rate = values[-1]
        else:
            for low, high, v in zip(boundaries[:-1], boundaries[1:],
                                    values[1:-1]):

                if step > low and step <= high:
                    learning_rate = v
                    break

        if self._summary:
            summary.scalar("learning_rate", learning_rate,
                           utils.get_global_step())

        return learning_rate

    def get_config(self):
        return {
            "boundaries": self._boundaries,
            "values": self._values,
        }


class LinearExponentialDecay(LearningRateSchedule):

    def __init__(self, learning_rate, warmup_steps, start_decay_step,
                 end_decay_step, n, summary=True):
        super(LinearExponentialDecay, self).__init__()

        self._learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._start_decay_step = start_decay_step
        self._end_decay_step = end_decay_step
        self._n = n
        self._summary = summary

    def __call__(self, step):
        # See reference: The Best of Both Worlds: Combining Recent Advances
        # in Neural Machine Translation
        n = self._n
        p = self._warmup_steps / n
        s = n * self._start_decay_step
        e = n * self._end_decay_step

        learning_rate = self._learning_rate

        learning_rate *= min(
            1.0 + (n - 1) * step / float(n * p),
            n,
            n * ((2 * n) ** (float(s - n * step) / float(e - s))))

        if self._summary:
            summary.scalar("learning_rate", learning_rate,
                           utils.get_global_step())

        return learning_rate

    def get_config(self):
        return {
            "learning_rate": self._learning_rate,
            "warmup_steps": self._warmup_steps,
            "start_decay_step": self._start_decay_step,
            "end_decay_step": self._end_decay_step,
        }
