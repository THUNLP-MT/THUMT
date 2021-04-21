# coding=utf-8
# Copyright 2017-Present The THUMT Authors

import abc
import torch

from collections.abc import Sequence
from torch.utils.data import IterableDataset
from thumt.data.iterator import Iterator
from thumt.data.vocab import Vocabulary
from thumt.tokenizers import Tokenizer
from typing import Any, Dict, NoReturn, List, Tuple, Union, Callable


class ElementSpec(object):

    def __init__(self, elem_type, shape):
        self._type = elem_type
        self._shape = shape

    def __repr__(self) -> str:
        return "%s, %s" % (self._type, self._shape)

    @property
    def elem_type(self) -> Any:
        return self._type

    @property
    def shape(self) -> str:
        return self._shape


class MapFunc(object):

    def __init__(self, fn: Callable, spec: ElementSpec):
        self._fn = fn
        self._elem_spec = spec

    def __call__(self, *args, **kwargs) -> Any:
        return self._fn(*args, **kwargs)

    @property
    def function(self):
        return self._fn

    @property
    def element_spec(self):
        return self._elem_spec


class Dataset(IterableDataset):

    def __init__(self):
        self._iterator = None

    def __iter__(self) -> Iterator:
        return Iterator(self)

    @abc.abstractproperty
    def _inputs(self) -> NoReturn:
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def copy(self) -> NoReturn:
        raise NotImplementedError("Dataset.copy not implemented.")

    @abc.abstractproperty
    def element_spec(self) -> NoReturn:
        raise NotImplementedError("Dataset.element_spec not implemented.")

    @property
    def name(self) -> str:
        return "Dataset"

    def new_iterator(self) -> Iterator:
        return Iterator(self)

    def background(self) -> "BackgroundDataset":
        return BackgroundDataset(self)

    def map(self, fn: MapFunc) -> "MapDataset":
        return MapDataset(self, fn)

    def padded_batch(self, batch_size: int, pad: int) -> "PaddedBatchDataset":
        return PaddedBatchDataset(self, batch_size, pad)

    def repeat(self, n: int) -> "RepeatDataset":
        return RepeatDataset(self, n)

    def shard(self, num_shards: int, index: int) -> "ShardDataset":
        return ShardDataset(self, num_shards, index)

    @abc.abstractmethod
    def set_inputs(self, datasets: Tuple["Dataset"]) -> NoReturn:
        raise NotImplementedError("Dataset.set_inputs not implemented.")

    def tokenize(self, tokenizer: Tokenizer, bos: bytes = b"<bos>",
                 eos: bytes = b"<eos>") -> "TokenizedLineDataset":
        return TokenizedLineDataset(self, tokenizer, bos, eos)

    @staticmethod
    def bucket_by_sequence_length(dataset: "Dataset",
                                  bucket_boundaries: List[int],
                                  batch_sizes: List[int],
                                  pad: int = 0,
                                  min_length: int = -1,
                                  max_length: int = 10000) -> "BucketDataset":
        return BucketDataset(dataset, bucket_boundaries, batch_sizes, pad,
                             min_length, max_length)

    @staticmethod
    def lookup(dataset: "Dataset", vocabulary: Dict[bytes, int], unk_id):
        return LookupDataset(dataset, vocabulary, unk_id)

    @staticmethod
    def zip(datasets: Tuple["Dataset"]) -> "ZipDataset":
        return ZipDataset(datasets)


class DatasetSource(Dataset):

    def _inputs(self):
        return []


class BackgroundDataset(Dataset):

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

        super(BackgroundDataset, self).__init__()

    def __repr__(self) -> str:
        return "<BackgroundDataset:%s>" % self._dataset

    def _inputs(self):
        return [self._dataset]

    @property
    def name(self):
        return "BackgroundDataset"

    @property
    def element_spec(self):
        return self._dataset._spec


class BucketDataset(Dataset):

    def __init__(self, dataset: Dataset, bucket_boundaries : List[int],
                 batch_sizes: List[int], pad: int = 0, min_length: int = -1,
                 max_length: int = 10000):
        if not self._check_type(dataset.element_spec):
            raise ValueError("The input dataset must produces an example of "
                             "`List[int]` or `Tuple[List[int], ...]`")

        self._dataset = dataset
        self._pad = pad
        self._bucket_boundaries = bucket_boundaries
        self._batch_sizes = batch_sizes
        self._min_length = min_length
        self._max_length = max_length

        _elem_spec = self._dataset.element_spec

        if _elem_spec.elem_type is List[int]:
            _elem_type = List[List[int]]
            _elem_shape = "[None, None]"
        else:
            # Tuple[List[int], ...] -> Tuple[List[List[int]], ...]
            args = _elem_spec.elem_type.__args__
            args = [List[t] for t in args]
            _elem_type = Tuple[tuple(args)]
            _elem_shape = ",".join(["[None, None]" for _ in args])

            if len(args) == 1:
                _elem_shape = "(" + _elem_shape + ",)"
            else:
                _elem_shape = "(" + _elem_shape + ")"

        self._spec = ElementSpec(_elem_type, _elem_shape)

        super(BucketDataset, self).__init__()

    def __repr__(self) -> str:
        return "<BucketDataset:%s>" % self._dataset

    def _inputs(self) -> List[Dataset]:
        return [self._dataset]

    def _check_type(self, elem_spec) -> bool:
        if elem_spec.elem_type is List[int]:
            return True
        elif not isinstance(elem_spec.elem_type,
                            type(Tuple[List[int], ...])):
            return False
        else:
            args = elem_spec.elem_type.__args__

            if len(args) == 0:
                return False

            for t in args:
                if t is not List[int]:
                    return False

            return True

    def copy(self) -> "BucketDataset":
        return BucketDataset(self._dataset.copy(), self._bucket_boundaries,
                             self._batch_sizes, self._pad)

    @property
    def name(self):
        return "BucketDataset"

    @property
    def bucket_boundaries(self) -> List[int]:
        return self._bucket_boundaries

    @property
    def batch_sizes(self) -> List[int]:
        return self._batch_sizes

    @property
    def min_length(self) -> int:
        return self._min_length

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def pad(self) -> bytes:
        return self._pad

    @property
    def element_spec(self) -> ElementSpec:
        return self._spec

    def set_inputs(self, datasets: Tuple[Dataset]) -> None:
        if len(datasets) != 1:
            raise ValueError("``datasets'' must be a tuple with one dataset.")

        dataset = datasets[0]

        if not self._check_type(dataset.element_spec):
            raise ValueError("The input dataset must produces an example of "
                             "`List[int]` or `Tuple[List[int], ...]`")

        self._dataset = dataset

        _elem_spec = self._dataset.element_spec

        if _elem_spec.elem_type is List[int]:
            _elem_type = List[List[int]]
            _elem_shape = "[None, None]"
        else:
            # Tuple[List[int], ...] -> Tuple[List[List[int]], ...]
            args = _elem_spec.elem_type.__args__
            args = [List[t] for t in args]
            _elem_type = Tuple[tuple(args)]
            _elem_shape = ",".join(["[None, None]" for _ in args])

            if len(args) == 1:
                _elem_shape = "(" + _elem_shape + ",)"
            else:
                _elem_shape = "(" + _elem_shape + ")"

        self._spec = ElementSpec(_elem_type, _elem_shape)


class FilterDataset(Dataset):

    def __init__(self, dataset: Dataset, min_len: int, max_len: int):
        if dataset.element_spec.elem_type is not List[int]:
            raise ValueError("The input dataset must produces an example of "
                             "`List[int]`.")

        self._dataset = dataset
        self._min_len = min_len
        self._max_len = max_len

        super(FilterDataset, self).__init__()

    def __repr__(self) -> str:
        return "<FilterDataset:%s>" % self._dataset

    def _inputs(self) -> List[Dataset]:
        return [self._dataset]

    def copy(self) -> "FilterDataset":
        return FilterDataset(self._dataset.copy(), self._min_len,
                             self._max_len)

    @property
    def name(self) -> str:
        return "FilterDataset"

    @property
    def max_len(self) -> int:
        return self._max_len

    @property
    def min_len(self) -> int:
        return self._min_len

    @property
    def element_spec(self) -> ElementSpec:
        return self._dataset._spec

    def set_inputs(self, datasets: Tuple[Dataset]) -> None:
        if len(datasets) != 1:
            raise ValueError("``datasets'' must be a tuple with one dataset.")

        self._dataset = datasets[0]


class LookupDataset(Dataset):

    def __init__(self, dataset: Dataset, vocabulary: Vocabulary,
                 unk_id : int = -1):
        if dataset.element_spec.elem_type is not List[bytes]:
            raise ValueError("The input dataset must produces an example of "
                             "`List[bytes]`.")
        self._dataset = dataset
        self._vocab = vocabulary
        self._unk_id = unk_id
        self._spec = ElementSpec(List[int], "[None]")
        super(LookupDataset, self).__init__()

    def __repr__(self) -> str:
        return "<LookupDataset:%s>" % self._dataset

    def _inputs(self) -> List[Dataset]:
        return [self._dataset]

    def copy(self) -> "LookupDataset":
        return LookupDataset(self._dataset.copy(), self._vocab, self._unk_id)

    @property
    def name(self) -> str:
        return "LookupDataset"

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocab

    @property
    def element_spec(self) -> ElementSpec:
        return self._spec

    def set_inputs(self, datasets: Tuple[Dataset]) -> None:
        if len(datasets) != 1:
            raise ValueError("``datasets'' must be a tuple with one dataset.")

        self._dataset = datasets[0]


class MapDataset(Dataset):

    def __init__(self, dataset: Dataset, fn: MapFunc):
        if not isinstance(fn, MapFunc):
            raise ValueError("fn must be an instance of MapFunc.")

        self._dataset = dataset
        self._fn = fn
        self._spec = fn.element_spec

        super(MapDataset, self).__init__()

    def __repr__(self) -> str:
        return "<MapDataset:%s>" % str(self._dataset)

    def copy(self) -> "MapDataset":
        return MapDataset(self._dataset, self._fn)

    @property
    def name(self) -> str:
        return "MapDataset"

    @property
    def element_spec(self) -> ElementSpec:
        return self._spec


class PaddedBatchDataset(Dataset):

    def __init__(self, dataset: Dataset, batch_size: int, pad: int):
        self._dataset = dataset
        self._batch_size = batch_size
        self._pad = pad

        _elem_spec = self._dataset.element_spec

        if _elem_spec.elem_type is List[int]:
            _elem_type = List[List[int]]
            _elem_shape = "[None, None]"
        else:
            # Tuple[List[int], ...] -> Tuple[List[List[int]], ...]
            args = _elem_spec.elem_type.__args__
            args = [List[t] for t in args]
            _elem_type = Tuple[tuple(args)]
            _elem_shape = ",".join(["[None, None]" for _ in args])

            if len(args) == 1:
                _elem_shape = "(" + _elem_shape + ",)"
            else:
                _elem_shape = "(" + _elem_shape + ")"

        self._spec = ElementSpec(_elem_type, _elem_shape)

        super(PaddedBatchDataset, self).__init__()

    def __repr__(self) -> str:
        return "<PaddedDataset:%s>" % str(self._dataset)

    def _inputs(self) -> List[Dataset]:
        return [self._dataset]

    def copy(self) -> "PaddedDataset":
        return PaddedBatchDataset(self._dataset.copy(),
                                  self._batch_size, self._pad_id)

    @property
    def name(self) -> str:
        return "PaddedBatchDataset"

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def pad(self) -> int:
        return self._pad

    @property
    def element_spec(self) -> ElementSpec:
        return self._spec


class RepeatDataset(Dataset):

    def __init__(self, dataset: Dataset, count: int = None):
        self._dataset = dataset
        self._count = -1 if count is None else count
        super(RepeatDataset, self).__init__()

    def __repr__(self) -> str:
        return "<RepeatDataset:%s,%d>" % (self._dataset, self._count)

    def _inputs(self) -> List[Dataset]:
        return [self._dataset]

    def copy(self) -> "RepeatDataset":
        return RepeatDataset(self._dataset.copy(), self._count)

    @property
    def name(self) -> str:
        return "RepeatDataset"

    @property
    def count(self) -> int:
        return self._count

    @property
    def element_spec(self) -> ElementSpec:
        self._dataset.element_spec

    def set_inputs(self, datasets: Tuple[Dataset]) -> None:
        if len(datasets) != 1:
            raise ValueError("``datasets'' must be a tuple with one dataset.")

        self._dataset = datasets[0]


class ShardDataset(Dataset):

    def __init__(self, dataset: Dataset, num_shards : int, index : int):
        self._dataset = dataset
        self._num_shards = num_shards
        self._index = index
        super(ShardDataset, self).__init__()

    def __repr__(self) -> str:
        info = (self._dataset, self._num_shards, self._index)
        return "<ShardDataset:%s,%d,%d>" % info

    def _inputs(self) -> List[Dataset]:
        return [self._dataset]

    def copy(self) -> "ShardDataset":
        return ShardDataset(self._dataset.copy(), self._num_shards,
                            self._index)

    @property
    def name(self) -> str:
        return "ShardDataset"

    @property
    def num_shards(self) -> int:
        return self._num_shards

    @property
    def index(self) -> int:
        return self._index

    @property
    def element_spec(self) -> ElementSpec:
        return self._dataset.element_spec

    def set_inputs(self, datasets: Tuple[Dataset]) -> None:
        if len(datasets) != 1:
            raise ValueError("``datasets'' must be a tuple with one dataset.")

        self._dataset = datasets[0]


class TextLineDataset(DatasetSource):

    def __init__(self, buffer_or_filename: Union[List, str]):
        self._source = buffer_or_filename
        self._spec = ElementSpec(bytes, "[]")
        super(TextLineDataset, self).__init__()

    def __repr__(self) -> str:
        return "<TextLineDataset:%s>" % self._filename

    def copy(self) -> "TextLineDataset":
        return TextLineDataset(self._source)

    @property
    def name(self) -> str:
        return "TextLineDataset"

    @property
    def input_source(self) -> Union[List, str]:
        return self._source

    @property
    def element_spec(self) -> ElementSpec:
        return self._spec

    def set_inputs(self, datasets: Tuple[Dataset]) -> None:
        return None


class TokenizedLineDataset(Dataset):

    def __init__(self, dataset: TextLineDataset, tokenizer: Tokenizer,
                 bos: bytes = b"<bos>", eos: bytes = b"<eos>"):
        elem_spec = dataset.element_spec

        if elem_spec.elem_type is not bytes or elem_spec.shape != "[]":
            raise ValueError("TokenizedLineDataset only accepts a dataset with"
                              " ElementSpec(bytes, '[None]')")

        self._dataset = dataset
        self._tokenizer = tokenizer
        self._bos = bos
        self._eos = eos
        self._spec = ElementSpec(List[bytes], "[None]")
        super(TokenizedLineDataset, self).__init__()

    def __repr__(self) -> str:
        return "<TokenizedLineDataset:%s>" % self._dataset

    def _inputs(self) -> List[Dataset]:
        return [self._dataset]

    def copy(self) -> "TokenizedLineDataset":
        return TokenizedLineDataset(self._dataset.copy(), self._tokenizer,
                                    self._bos, self._eos)

    @property
    def name(self) -> str:
        return "TokenizedLineDataset"

    @property
    def element_spec(self) -> ElementSpec:
        return self._spec

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def bos(self) -> bytes:
        return self._bos

    @property
    def eos(self) -> bytes:
        return self._eos

    def set_inputs(self, datasets: Tuple[Dataset]) -> None:
        if len(datasets) != 1:
            raise ValueError("``datasets'' must be a tuple with one dataset.")

        self._dataset = datasets[0]


class ZipDataset(Dataset):

    def __init__(self, datasets: Tuple[Dataset]):
        if not isinstance(datasets, tuple):
            raise ValueError("ZipDataset expects a tuple of datasets as "
                             "the input.")

        self._datasets = datasets
        self._num_inputs = len(datasets)

        _type = tuple(dataset.element_spec.elem_type for dataset in datasets)
        _type = Tuple[_type]
        _shape = ",".join([dataset.element_spec.shape for dataset in datasets])

        if len(self._datasets) == 1:
            _shape = "(" + _shape + ",)"
        else:
            _shape = "(" + _shape + ",)"

        self._spec = ElementSpec(_type, _shape)

        super(ZipDataset, self).__init__()

    def __repr__(self) -> str:
        if len(self._datasets == 1):
            ds_repr = "(%s,)" % self._datasets[0]
        else:
            ds_repr = ",".join([str(ds) for ds in self._datasets])
        return "<ZipDataset:(%s,)>" % ds_repr

    def _inputs(self) -> List[Dataset]:
        return list(self._datasets)

    def copy(self) -> "ZipDataset":
        datasets = tuple([ds.copy() for ds in self._datasets])
        return ZipDataset(datasets)

    @property
    def name(self) -> str:
        return "ZipDataset"

    @property
    def num_inputs(self) -> int:
        return self._num_inputs

    @property
    def element_spec(self) -> ElementSpec:
        return self._spec

    def set_inputs(self, datasets: Tuple[Dataset]) -> None:
        self._datasets = datasets

        _type = tuple(dataset.element_spec.elem_type for dataset in datasets)
        _type = Tuple[_type]
        _shape = ",".join([dataset.element_spec.shape for dataset in datasets])

        if len(self._datasets) == 1:
            _shape = "(" + _shape + ",)"
        else:
            _shape = "(" + _shape + ",)"

        self._spec = ElementSpec(_type, _shape)
