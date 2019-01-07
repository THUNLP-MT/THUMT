#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import sys

if __name__ == "__main__":
    with open(sys.argv[1]) as fd:
        voc = pickle.load(fd)

    ivoc = {}

    for key in voc:
        ivoc[voc[key]] = key

    with open(sys.argv[2], "w") as fd:
        for key in ivoc:
            val = ivoc[key]
            fd.write(val + "\n")
