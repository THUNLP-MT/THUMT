#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import sys


def _open(filename, mode="r", encoding="utf-8"):
    if sys.version_info.major == 2:
        return open(filename, mode=mode)
    elif sys.version_info.major == 3:
        return open(filename, mode=mode, encoding=encoding)
    else:
        raise RuntimeError("Unknown Python version for running!")


if __name__ == "__main__":
    with _open(sys.argv[1]) as fd:
        voc = pickle.load(fd)

    ivoc = {}

    for key in voc:
        ivoc[voc[key]] = key

    with _open(sys.argv[2], "w") as fd:
        for key in ivoc:
            val = ivoc[key]
            fd.write(val + "\n")

