# coding=utf-8
# Copyright 2017-Present The THUMT Authors

import numpy as np
import six
import torch

from typing import Union


class Vocabulary(object):

    def __init__(self, filename):
        self._idx2word = {}
        self._word2idx = {}
        cnt = 0

        with open(filename, "rb") as fd:
            for line in fd:
                self._word2idx[line.strip()] = cnt
                self._idx2word[cnt] = line.strip()
                cnt = cnt + 1

    def __getitem__(self, key: Union[bytes, int]):
        if isinstance(key, int):
            return self._idx2word[key]
        elif isinstance(key, bytes):
            return self._word2idx[key]
        elif isinstance(key, str):
            key = key.encode("utf-8")
            return self._word2idx[key]
        else:
            raise LookupError("Cannot lookup key %s." % key)

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.encode("utf-8")

        return key in self._word2idx

    def __iter__(self):
        return six.iterkeys(self._word2idx)

    def __len__(self):
        return len(self._idx2word)
