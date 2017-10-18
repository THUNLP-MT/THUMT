#!/usr/bin/env python
# coding=utf-8
# Copyright 2017 The THUMT Authors

import sys
import cPickle


if __name__ == "__main__":
    with open(sys.argv[1]) as fd:
        voc = cPickle.load(fd)

    ivoc = {}

    for key in voc:
        ivoc[voc[key]] = key

    with open(sys.argv[2], "w") as fd:
        for key in ivoc:
            val = ivoc[key]
            fd.write(val + "\n")
