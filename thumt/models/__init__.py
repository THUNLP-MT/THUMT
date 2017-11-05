# coding=utf-8
# Copyright 2017 The THUMT Authors

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.transformer


def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return thumt.models.rnnsearch.RNNsearch
    elif name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    else:
        raise LookupError("Unknown model %s" % name)
