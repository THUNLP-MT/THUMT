import abc
import json
import base64
import collections
import regex as re

from typing import List, NoReturn
from thumt.tokenizers.tokenizer import Tokenizer


_RULES = [
    # Open/Initial puncutation
    [
        ("([\\p{Ps}\\p{Pi}])(.)", "\\1 \\2"),
        ("([\\p{Ps}\\p{Pi}]) (.)", "\\1\\2")
    ],
    # Close/Final puncutation
    [
        ("(.)([\\p{Pe}\\p{Pf}])", "\\1 \\2"),
        ("(.) ([\\p{Pe}\\p{Pf}])", "\\1\\2")
    ],
    # Tokenize the following symbols
    [
        ("([|~\\\\^_`#&*+<=>@/\\-])", " \\1 "),
        ("[ ]?([|~\\\\^_`#&*+<=>@/\\-])[ ]?", "\\1"),
    ],
    # Tokenize colon
    [
        ("([\\p{L}]): ", "\\1 : "),
        ("([\\p{L}]) : ", "\\1: ")
    ],
    # Tokenize period and comma
    [
        ("(.)([\\.,!?;]) ", "\\1 \\2 "),
        ("(.) ([\\.,!?;]) ", "\\1\\2 "),
    ],
    # Tokenize period and comma at end of the input
    [
        ("(.)([\\.,!?;])$", "\\1 \\2"),
        ("(.) ([\\.,!?;])$", "\\1\\2"),
    ],
    # Tokenize quotation mark
    [
        ("([\\p{L}])\"([\\p{L}])", "\\1 <quot> \\2"),
        ("([\\p{L}]) <quot> ([\\p{L}])", "\\1\"\\2"),
    ],
    [
        ("\"([\\p{L}\\p{N}])", "<lquot> \\1"),
        ("<lquot> ([\\p{L}\\p{N}])", "\"\\1"),
    ],
    [
        ("([\\p{L}\\p{N}])\"", "\\1 <rquot>"),
        ("([\\p{L}\\p{N}]) <rquot>", "\\1\""),
    ],
    # Tokenize Apostrophe
    [
        ("([\\p{L}])'([\\p{L}])", "\\1 <apos> \\2"),
        ("([\\p{L}]) <apos> ([\\p{L}])", "\\1'\\2"),
    ],
    [
        ("'([\\p{L}])", "<lapos> \\1"),
        ("<lapos> ([\\p{L}])", "\"\\1"),
    ],
    [
        ("([\\p{L}])'", "\\1 <rapos>"),
        ("([\\p{L}]) <rapos>", "\\1\""),
    ],
    # Replace control/separators with space
    [
        ("[\\p{C}\\p{Z}]+", " ")
    ],
    # Remove starting space
    [
        ("^ (.*)", "\\1")
    ],
    # Remove trailing space
    [
        ("(.*) $", "\\1")
    ]
]

_TOKEN_PATTERNS = [re.compile(rule[0][0]) for rule in _RULES]
_TOKEN_REPL = [rule[0][1] for rule in _RULES]

_DETOKEN_PATTERNS = [
    re.compile(rule[1][0]) if len(rule) == 2 else None for rule in _RULES
][::-1]
_DETOKEN_REPL = [
    rule[1][1] if len(rule) == 2 else None for rule in _RULES
][::-1]


class UnicodeTokenizer(Tokenizer):

    def __init__(self, name="unicode_tokenizer"):
        super(UnicodeTokenizer, self).__init__()

    def encode(self, inp: bytes) -> List[bytes]:
        inp_str = inp
        for pat, repl in zip(_TOKEN_PATTERNS, _TOKEN_REPL):
            input_str = re.sub(pat, repl, input_str)

        return input_str

    def decode(self, inp: List[bytes]) -> bytes:
        input_str = b" ".join(inp)

        for pat, repl in zip(_DETOKEN_PATTERNS, _DETOKEN_REPL):
            if not pat:
                continue
            input_str = re.sub(pat, repl, input_str)

        return input_str
