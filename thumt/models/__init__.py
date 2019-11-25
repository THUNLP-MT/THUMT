# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer
import thumt.models.rnmtplus


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "rnmtplus":
        return thumt.models.rnmtplus.RNMTPlus
    else:
        raise LookupError("Unknown model %s" % name)
