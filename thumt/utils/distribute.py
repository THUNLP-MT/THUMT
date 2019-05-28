# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

_ENGINE = None


def enable_distributed_training():
    global _ENGINE
    try:
        import horovod.tensorflow as hvd
        _ENGINE = hvd
        hvd.init()
    except ImportError:
        sys.stderr.write("Error: You must install horovod first in order to"
                         " enable distributed training.\n")
        exit()


def is_distributed_training_mode():
    return _ENGINE is not None


def rank():
    return _ENGINE.rank() if _ENGINE is not None else 0


def local_rank():
    return _ENGINE.local_rank() if _ENGINE is not None else 0


def size():
    return _ENGINE.size() if _ENGINE is not None else 1


def all_reduce(tensor):
    if _ENGINE is None:
        return tensor

    return _ENGINE.allreduce(tensor, compression=_ENGINE.Compression.fp16)


def get_broadcast_hook():
    if not _ENGINE:
        return None
    return _ENGINE.BroadcastGlobalVariablesHook(0)
