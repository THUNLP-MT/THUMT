import abc

from typing import List, NoReturn


class Tokenizer(object):

    def __init__(self, name: str):
        self._name = name

    @abc.abstractmethod
    def __repr__(self) -> NoReturn:
        raise NotImplementedError("Tokenizer.__repr__ not implemented.")

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def encode(self, inp: bytes) -> NoReturn:
        raise NotImplementedError("Tokenizer.encode not implemented.")

    @abc.abstractmethod
    def decode(self, inp: List[bytes]) -> NoReturn:
        raise NotImplementedError("Tokenizer.decode not implemented.")


class WhiteSpaceTokenizer(Tokenizer):

    def __init__(self):
        super(WhiteSpaceTokenizer, self).__init__("WhiteSpaceTokenizer")

    def __repr__(self) -> str:
        return "WhiteSpaceTokenizer()"

    def encode(self, inp: bytes) -> List[bytes]:
        return inp.strip().split()

    def decode(self, inp: List[bytes]) -> bytes:
        return b" ".join(inp)
