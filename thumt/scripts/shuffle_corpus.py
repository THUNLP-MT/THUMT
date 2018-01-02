# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy


def parseargs():
    parser = argparse.ArgumentParser(description="Shuffle corpus")

    parser.add_argument("--corpus", nargs="+", required=True,
                        help="input corpora")
    parser.add_argument("--suffix", type=str, default="shuf",
                        help="Suffix of output files")
    parser.add_argument("--seed", type=int, help="Random seed")

    return parser.parse_args()


def main(args):
    name = args.corpus
    suffix = "." + args.suffix
    stream = [open(item, "r") for item in name]
    data = [fd.readlines() for fd in stream]
    minlen = min([len(lines) for lines in data])

    if args.seed:
        numpy.random.seed(args.seed)

    indices = numpy.arange(minlen)
    numpy.random.shuffle(indices)

    newstream = [open(item + suffix, "w") for item in name]

    for idx in indices.tolist():
        lines = [item[idx] for item in data]

        for line, fd in zip(lines, newstream):
            fd.write(line)

    for fdr, fdw in zip(stream, newstream):
        fdr.close()
        fdw.close()


if __name__ == "__main__":
    parsed_args = parseargs()
    main(parsed_args)
