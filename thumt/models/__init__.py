# coding=utf-8
# Copyright 2017 The THUMT Authors

import thumt.models.rnnsearch


def get_model(name):
    if name == "rnnsearch":
        return thumt.models.rnnsearch.RNNsearch
    else:
        raise LookupError("Unknown model %s" % name)
