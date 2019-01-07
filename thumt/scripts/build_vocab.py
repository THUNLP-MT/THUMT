#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections


def count_words(filename):
    counter = collections.Counter()

    with open(filename, "r") as fd:
        for line in fd:
            words = line.strip().split()
            counter.update(words)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))

    return words, counts


def control_symbols(string):
    if not string:
        return []
    else:
        return string.strip().split(",")


def save_vocab(name, vocab):
    if name.split(".")[-1] != "txt":
        name = name + ".txt"

    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words, ids = list(zip(*pairs))

    with open(name, "w") as f:
        for word in words:
            f.write(word + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Create vocabulary")

    parser.add_argument("corpus", help="input corpus")
    parser.add_argument("output", default="vocab.txt",
                        help="Output vocabulary name")
    parser.add_argument("--limit", default=0, type=int, help="Vocabulary size")
    parser.add_argument("--control", type=str, default="<pad>,<eos>,<unk>",
                        help="Add control symbols to vocabulary. "
                             "Control symbols are separated by comma.")

    return parser.parse_args()


def main(args):
    vocab = {}
    limit = args.limit
    count = 0

    words, counts = count_words(args.corpus)
    ctrl_symbols = control_symbols(args.control)

    for sym in ctrl_symbols:
        vocab[sym] = len(vocab)

    for word, freq in zip(words, counts):
        if limit and len(vocab) >= limit:
            break

        if word in vocab:
            print("Warning: found duplicate token %s, ignored" % word)
            continue

        vocab[word] = len(vocab)
        count += freq

    save_vocab(args.output, vocab)

    print("Total words: %d" % sum(counts))
    print("Unique words: %d" % len(words))
    print("Vocabulary coverage: %4.2f%%" % (100.0 * count / sum(counts)))


if __name__ == "__main__":
    main(parse_args())
