#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import argparse
import collections
import torch
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Create vocabulary")

    parser.add_argument("--path", help="checkpoint directory")
    parser.add_argument("--output", default="average",
                        help="Output path")
    parser.add_argument("--checkpoints", default=5, type=int,
                        help="Number of checkpoints to average")

    return parser.parse_args()


def list_checkpoints(path):
    names = glob.glob(os.path.join(path, "*.pt"))

    if not names:
        return None

    vals = []

    for name in names:
        counter = int(name.rstrip(".pt").split("-")[-1])
        vals.append([counter, name])

    return [item[1] for item in sorted(vals)]


def main(args):
    checkpoints = list_checkpoints(args.path)

    if not checkpoints:
        raise ValueError("No checkpoint to average")

    checkpoints = checkpoints[-args.checkpoints:]
    values = collections.OrderedDict()

    for checkpoint in checkpoints:
        print("Loading checkpoint: %s" % checkpoint)
        state = torch.load(checkpoint, map_location="cpu")["model"]

        for key in state:
            if key not in values:
                values[key] = state[key].float().clone()
            else:
                values[key].add_(state[key].float())

    for key in values:
        values[key].div_(len(checkpoints))

    state = {"step": 0, "epoch": 0, "model": values}

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    torch.save(state, os.path.join(args.output, "average-0.pt"))
    params_pattern = os.path.join(args.path, "*.json")
    params_files = glob.glob(params_pattern)

    for name in params_files:
        new_name = name.replace(args.path.rstrip("/"), args.output.rstrip("/"))
        shutil.copy(name, new_name)


if __name__ == "__main__":
    main(parse_args())
