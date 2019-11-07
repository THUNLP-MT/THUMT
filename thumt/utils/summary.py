# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.distributed as dist
import torch.utils.tensorboard as tensorboard

_SUMMARY_WRITER = None


def init(log_dir, enable=True):
    global _SUMMARY_WRITER

    if enable and dist.get_rank() == 0:
        _SUMMARY_WRITER = tensorboard.SummaryWriter(log_dir)


def scalar(tag, scalar_value, global_step=None, walltime=None,
           write_every_n_steps=100):
    if _SUMMARY_WRITER is not None:
        if global_step % write_every_n_steps == 0:
            _SUMMARY_WRITER.add_scalar(tag, scalar_value, global_step,
                                       walltime)


def histogram(tag, values, global_step=None, bins="tensorflow", walltime=None,
              max_bins=None, write_every_n_steps=100):
    if _SUMMARY_WRITER is not None:
        if global_step % write_every_n_steps == 0:
            _SUMMARY_WRITER.add_histogram(tag, values, global_step, bins,
                                          walltime, max_bins)

def close():
    if _SUMMARY_WRITER is not None:
        _SUMMARY_WRITER.close()
