# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import queue
import threading
import torch

import torch.distributed as dist
import torch.utils.tensorboard as tensorboard

_SUMMARY_WRITER = None
_QUEUE = None
_THREAD = None


class SummaryWorker(threading.Thread):

    def run(self):
        global _QUEUE

        while True:
            item = _QUEUE.get()
            name, kwargs = item

            if name == "stop":
                break

            self.write_summary(name, **kwargs)

    def write_summary(self, name, **kwargs):
        if name == "scalar":
            _SUMMARY_WRITER.add_scalar(**kwargs)
        elif name == "histogram":
            _SUMMARY_WRITER.add_histogram(**kwargs)

    def stop(self):
        global _QUEUE
        _QUEUE.put(("stop", None))
        self.join()


def init(log_dir, enable=True):
    global _SUMMARY_WRITER
    global _QUEUE
    global _THREAD

    if enable and dist.get_rank() == 0:
        _SUMMARY_WRITER = tensorboard.SummaryWriter(log_dir)
        _QUEUE = queue.Queue()
        thread = SummaryWorker(daemon=True)
        thread.start()
        _THREAD = thread


def scalar(tag, scalar_value, global_step=None, walltime=None,
           write_every_n_steps=100):

    if _SUMMARY_WRITER is not None:
        if global_step % write_every_n_steps == 0:
            scalar_value = float(scalar_value)
            kwargs = dict(tag=tag, scalar_value=scalar_value,
                          global_step=global_step, walltime=walltime)
            _QUEUE.put(("scalar", kwargs))


def histogram(tag, values, global_step=None, bins="tensorflow", walltime=None,
              max_bins=None, write_every_n_steps=100):

    if _SUMMARY_WRITER is not None:
        if global_step % write_every_n_steps == 0:
            values = values.detach().cpu()
            kwargs = dict(tag=tag, values=values, global_step=global_step,
                          bins=bins, walltime=walltime, max_bins=max_bins)
            _QUEUE.put(("histogram", kwargs))


def close():
    if _SUMMARY_WRITER is not None:
        _THREAD.stop()
        _SUMMARY_WRITER.close()
