# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import tensorflow as tf
import numpy


def sparse_softmax_cross_entropy_with_logits(labels, logits):
    if logits.dim() < 1:
        raise ValueError("logits must not be scalars.")

    if labels.dim() != logits.dim() - 1:
        raise ValueError("The rank of the labels is not equal to the rank "
                         "of the logits minus one.")

    shape = labels.shape
    logits = torch.reshape(logits, [-1, logits.shape[-1]])
    labels = torch.reshape(labels, [-1])

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    batch_idx = torch.arange(labels.shape[0], device=labels.device)
    outputs = log_probs[batch_idx, labels]
    return -torch.reshape(outputs, shape)


def smoothed_softmax_cross_entropy_with_logits(labels, logits, smoothing=0.0):
    if not smoothing:
        return sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)

    # label smoothing
    n = logits.shape[-1] - 1.0
    p = 1.0 - smoothing
    q = smoothing / n

    shape = labels.shape
    logits = torch.reshape(logits, [-1, logits.shape[-1]])
    labels = torch.reshape(labels, [-1])

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    sum_probs = torch.sum(log_probs, dim=-1)
    batch_idx = torch.arange(labels.shape[0], device=labels.device)
    selected_probs = log_probs[batch_idx, labels]
    loss = p * selected_probs + q * (sum_probs - selected_probs)
    loss = -torch.reshape(loss, shape)
    normalizing = -(p * math.log(p) + n * q * math.log(q + 1e-20))

    return loss - normalizing
