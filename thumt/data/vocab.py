# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


def _lookup(x, vocab):
    x = x.tolist()
    y = []

    for i, batch in enumerate(x):
        ids = []
        for j, v in enumerate(batch):
            ids.append(vocab[v] if v in vocab else 2)
        y.append(ids)

    return torch.LongTensor(np.array(y, dtype="int32")).cuda()


def load_vocabulary(filename):
    vocab = []
    with open(filename, "rb") as fd:
        for line in fd:
            vocab.append(line.strip())

    word2idx = {}
    idx2word = {}

    for idx, word in enumerate(vocab):
        word2idx[word] = idx
        idx2word[idx] = word

    return vocab, word2idx, idx2word


def lookup(features, mode, params):
    if mode != "infer":
        inputs, labels = features
        source, target = inputs
        source = source.numpy()
        target = target.numpy()
        labels = labels.numpy()

        source = _lookup(source, params.lookup["source"])
        target = _lookup(target, params.lookup["target"])
        labels = _lookup(labels, params.lookup["target"])

        features = {
            "source": source, "target": target, "labels": labels
        }

        return features
    else:
        source = features.numpy()
        source = _lookup(source, params.lookup["source"])

        features = {
            "source": source
        }

        return features
