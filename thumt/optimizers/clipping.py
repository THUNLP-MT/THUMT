# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


def global_norm_clipper(value):
    def clip_fn(gradients, grad_norm):
        if not float(value) or grad_norm < value:
            return False, gradients

        scale = value / grad_norm

        gradients = [grad.data.mul_(scale)
            if grad is not None else None for grad in gradients]

        return False, gradients

    return clip_fn


def value_clipper(clip_min, clip_max):
    def clip_fn(gradients, grad_norm):
        gradients = [
            grad.data.clamp_(clip_min, clip_max)
            if grad is not None else None for grad in gradients]

        return False, None

    return clip_fn


def adaptive_clipper(rho):
    norm_avg = 0.0
    norm_stddev = 0.0
    log_norm_avg = 0.0
    log_norm_sqr = 0.0

    def clip_fn(gradients, grad_norm):
        nonlocal norm_avg
        nonlocal norm_stddev
        nonlocal log_norm_avg
        nonlocal log_norm_sqr

        norm = grad_norm
        log_norm = math.log(norm)

        avg = rho * norm_avg + (1.0 - rho) * norm
        log_avg = rho * log_norm_avg + (1.0 - rho) * log_norm
        log_sqr = rho * log_norm_sqr + (1.0 - rho) * (log_norm ** 2)
        stddev = (log_sqr - (log_avg ** 2)) ** -0.5

        norm_avg = avg
        log_norm_avg = log_avg
        log_norm_sqr = log_sqr
        norm_stddev = rho * stddev + (1.0 - rho) * stddev

        reject = False

        if norm > norm_avg + 4 * math.exp(norm_stddev):
            reject = True

        return reject, gradients

    return clip_fn
