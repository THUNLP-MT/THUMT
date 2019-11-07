# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch


class SmoothedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, smoothing=0.0, normalize=True):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.normalize = normalize

    def forward(self, logits, labels):
        shape = labels.shape
        logits = torch.reshape(logits, [-1, logits.shape[-1]])
        labels = torch.reshape(labels, [-1])

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        batch_idx = torch.arange(labels.shape[0], device=logits.device)
        loss = log_probs[batch_idx, labels]

        if not self.smoothing:
            return -torch.reshape(loss, shape)

        n = logits.shape[-1] - 1.0
        p = 1.0 - self.smoothing
        q = self.smoothing / n

        if log_probs.dtype != torch.float16:
            sum_probs = torch.sum(log_probs, dim=-1)
            loss = p * loss + q * (sum_probs - loss)
        else:
            # Prevent FP16 overflow
            sum_probs = torch.sum(log_probs.to(torch.float32), dim=-1)
            loss = loss.to(torch.float32)
            loss = p * loss + q * (sum_probs - loss)
            loss = loss.to(torch.float16)

        loss = -torch.reshape(loss, shape)

        if self.normalize:
            normalizing = -(p * math.log(p) + n * q * math.log(q + 1e-20))
            return loss - normalizing
        else:
            return loss
