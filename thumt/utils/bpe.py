# coding=utf-8
# Copyright 2017-2020 The THUMT Authors
# Modified from subword-nmt

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re


class BPE(object):

    def __init__(self, bpe_path, merges=-1, separator="@@"):
        with open(bpe_path, "r", encoding="utf-8") as fd:
            firstline = fd.readline()

            if not firstline.startswith("#version:"):
                raise ValueError("THUMT only support BPE version >= 0.2.")

            codes = tuple([item.strip("\r\n").split(" ")
                           for (n, item) in enumerate(fd)
                           if (n < merges or merges == -1)])

        for _, item in enumerate(codes):
            if len(item) != 2:
                raise ValueError("Error: invalid BPE codes found.")

        self._codes = {}

        for (i, code) in enumerate(codes):
            if tuple(code) not in self._codes:
                self._codes[tuple(code)] = i

        self._separator = separator

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]

        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char

        return pairs


    def _encode_word(self, orig):
        word = tuple(orig[:-1]) + (orig[-1] + "</w>",)
        pairs = self._get_pairs(word)

        if not pairs:
            return (orig,)

        while True:
            bigram = min(pairs, key=lambda x: self._codes.get(x, float("inf")))

            if bigram not in self._codes:
                break

            first, second = bigram
            new_word = []

            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and word[i + 1] == second:
                    if i < len(word) - 1:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word

            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)

        if word[-1] == "</w>":
            word = word[:-1]
        elif word[-1].endswith("</w>"):
            word = word[:-1] + (word[-1].replace("</w>", ""),)

        return word

    def encode(self, s):
        words = s.strip().split()
        output = []

        for word in words:
            if not word:
                continue

            new_word = self._encode_word(word)

            for item in new_word[:-1]:
                output.append(item + self._separator)

            output.append(new_word[-1])

        return output

    @staticmethod
    def decode(s):
        if isinstance(s, str):
            return re.sub("(@@ )|(@@ ?$)", "", s)
        else:
            return re.sub(b"(@@ )|(@@ ?$)", b"", s)
