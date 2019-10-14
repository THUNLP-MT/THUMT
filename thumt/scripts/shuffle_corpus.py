# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy
import sys


def _open(filename, mode="r", encoding="utf-8"):
    if sys.version_info.major == 2:
        return open(filename, mode=mode)
    elif sys.version_info.major == 3:
        return open(filename, mode=mode, encoding=encoding)
    else:
        raise RuntimeError("Unknown Python version for running!")


def parseargs():
    parser = argparse.ArgumentParser(description="Shuffle corpus")

    parser.add_argument("--corpus", nargs="+", required=True,
                        help="input corpora")
    parser.add_argument("--suffix", type=str, default="shuf",
                        help="Suffix of output files")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="shard number")

    return parser.parse_args()


def main(args):
    name = args.corpus
    suffix = "." + args.suffix
    stream = [_open(item, "r") for item in name]
    data = [fd.readlines() for fd in stream]
    minlen = min([len(lines) for lines in data])
    count = 0

    if args.seed:
        numpy.random.seed(args.seed)

    indices = numpy.arange(minlen)
    numpy.random.shuffle(indices)

    if args.num_shards == 1:
        newstream = [[_open(item + suffix, "w") for item in name]]
    else:
        newstream = [[_open(item + "-%s-of-%s" % (i, args.num_shards), "w")
                      for item in name] for i in range(args.num_shards)]

    for idx in indices.tolist():
        lines = [item[idx] for item in data]

        for line, fd in zip(lines, newstream[count % args.num_shards]):
            fd.write(line)

        count += 1

    for fdr in stream:
        fdr.close()

    for fds in newstream:
        for fd in fds:
            fd.close()


if __name__ == "__main__":
    parsed_args = parseargs()
    main(parsed_args)
