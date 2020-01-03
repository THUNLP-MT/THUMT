#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import torch


def convert_tensor(variables, name, tensor):
    # 1. replace '/' with '.'
    name = name.replace("/", ".")
    # 2. strip "transformer."
    if "transformer" in name:
        name = name[12:]
    # 3. layer_* -> layers.*
    name = name.replace("layer_", "layers.")
    name = name.replace("layers.norm", "layer_norm")
    # 4. offset -> bias
    name = name.replace("offset", "bias")
    # 5. scale -> weight
    name = name.replace("scale", "weight")
    # 6. matrix -> weight, transpose
    if "matrix" in name:
        name = name.replace("matrix", "weight")
        tensor = tensor.transpose()
    # 7. multihead_attention -> attention
    name = name.replace("multihead_attention", "attention")
    variables[name] = torch.tensor(tensor)


def main():
    if len(sys.argv) != 3:
        print("convert_checkpoint.py input output")
        exit(-1)

    var_list = tf.train.list_variables(sys.argv[1])
    variables = {}
    reader = tf.train.load_checkpoint(sys.argv[1])

    for (name, _) in var_list:
        tensor = reader.get_tensor(name)
        if not name.startswith("transformer") or "Adam" in name:
            continue

        if "qkv_transform" in name:
            if "matrix" in name:
                n1 = name.replace("qkv_transform", "q_transform")
                n2 = name.replace("qkv_transform", "k_transform")
                n3 = name.replace("qkv_transform", "v_transform")
                v1, v2, v3 = np.split(tensor, 3, axis=1)
                convert_tensor(variables, n1, v1)
                convert_tensor(variables, n2, v2)
                convert_tensor(variables, n3, v3)
            elif "bias" in name:
                n1 = name.replace("qkv_transform", "q_transform")
                n2 = name.replace("qkv_transform", "k_transform")
                n3 = name.replace("qkv_transform", "v_transform")
                v1, v2, v3 = np.split(tensor, 3)
                convert_tensor(variables, n1, v1)
                convert_tensor(variables, n2, v2)
                convert_tensor(variables, n3, v3)
        elif "kv_transform" in name:
            if "matrix" in name:
                n1 = name.replace("kv_transform", "k_transform")
                n2 = name.replace("kv_transform", "v_transform")
                v1, v2 = np.split(tensor, 2, axis=1)
                convert_tensor(variables, n1, v1)
                convert_tensor(variables, n2, v2)
            elif "bias" in name:
                n1 = name.replace("kv_transform", "k_transform")
                n2 = name.replace("kv_transform", "v_transform")
                v1, v2 = np.split(tensor, 2)
                convert_tensor(variables, n1, v1)
                convert_tensor(variables, n2, v2)
        elif "multihead_attention/output_transform" in name:
            name = name.replace("multihead_attention/output_transform",
                                "multihead_attention/o_transform")
            convert_tensor(variables, name, tensor)
        elif "ffn_layer/output_layer/linear" in name:
            name = name.replace("ffn_layer/output_layer/linear",
                                "ffn_layer/output_transform")
            convert_tensor(variables, name, tensor)
        elif "ffn_layer/input_layer/linear" in name:
            name = name.replace("ffn_layer/input_layer/linear",
                                "ffn_layer/input_transform")
            convert_tensor(variables, name, tensor)
        else:
            convert_tensor(variables, name, tensor)

    torch.save({"model": variables}, sys.argv[2])


if __name__ == "__main__":
    main()
