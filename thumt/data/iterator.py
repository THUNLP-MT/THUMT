# coding=utf-8
# Copyright 2017-Present The THUMT Authors

import abc
import time
import queue
import threading

from typing import Any, Dict, List, NoReturn, Tuple, Union


def _profile(msg: str, enable: bool = True):
    def decorator(func):
        def on_call(*args, **kwargs):
            start_time = time.perf_counter()
            ret = func(*args, **kwargs)

            if enable:
                print(msg, time.perf_counter() - start_time)
            return ret

        return on_call

    return decorator


def _maybe_to_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def _unzip(x):
    return list(zip(*x))


class _FileWrapper(object):

    def __init__(self, buffer: List):
        self._buffer = buffer
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._buffer):
            raise StopIteration

        line = self._buffer[self._index]
        self._index += 1

        return line

    def readline(self):
        try:
            line = self._buffer[self._index]
            self._index += 1
        except:
            line = ""
        return line

    def readlines(self):
        return self._buffer

    def seek(self, offset: int):
        self._index = offset

    def tell(self):
        return self._index


class _DatasetWorker(threading.Thread):

    def init(self, dataset: "Dataset", id: int = 0, buffer_size: int = 64):
        self._iterator = iter(dataset)
        self._buffer = queue.Queue(buffer_size)
        self._buffer_size = buffer_size
        self._empty = False
        self._id = id

    def get(self) -> Any:
        if self._empty and self._buffer.empty():
            return None

        return self._buffer.get()

    def run(self) -> None:
        while True:
            try:
                self._buffer.put(next(self._iterator))
            except StopIteration:
                break

        self._empty = True

    def is_empty(self) -> bool:
        return self._empty


class IteratorBase(object):

    def __init__(self):
        pass

    def __iter__(self) -> "IteratorBase":
        return self

    def state(self) -> Dict:
        return {}

    @abc.abstractmethod
    def __next__(self) -> NoReturn:
        raise NotImplementedError("IteratorBase.__next__ not implemented.")


class _BackgroundDSIter(IteratorBase):

    def __init__(self, dataset: "BackgroundDataset"):
        self._thread = _DatasetWorker(daemon=True)
        self._thread.init(dataset._dataset)
        self._thread.start()

    def __next__(self) -> Any:
        item = self._thread.get()

        if item is None:
            self._thread.join()
            raise StopIteration

        return item


class _BucketDSIter(IteratorBase):

    def __init__(self, dataset: "BucketDataset"):
        self._pad = dataset.pad
        self._bucket_boundaries = dataset.bucket_boundaries
        self._batch_sizes = dataset.batch_sizes
        self._iterator = iter(dataset._dataset)
        self._spec = dataset.element_spec
        self._buckets = [[] for _ in dataset.batch_sizes]
        self._priority = [k for k in range(len(dataset.batch_sizes))]
        self._min_length = dataset.min_length
        self._max_length = dataset.max_length
        self._max_fill = max(dataset.batch_sizes)
        self._bucket_map = {}

        # length to bucket index
        max_len = max(dataset.bucket_boundaries)
        idx = 0

        # [0, max_boundary]
        for i in range(0, max_len + 1):
            for idx in range(len(self._bucket_boundaries)):
                if i <= self._bucket_boundaries[idx]:
                    self._bucket_map[i] = idx
                    break

        super(_BucketDSIter, self).__init__()


    def __iter__(self) -> "_BucketDSIter":
        return self

    @_profile("_BucketDSIter", False)
    def __next__(self) -> Union[List[List[int]],
                                Tuple[List[List[int]], ...]]:
        try:
            while True:
                idx = self._get_bucket()

                if idx >= 0:
                    return self._get_content(idx)
                else:
                    self._fill()
        except StopIteration:
            idx = self._get_nonempty_bucket()

            if idx < 0:
                raise StopIteration

            return self._get_content(idx)

    @_profile("_BucketDSIter_fill", False)
    def _fill(self) -> None:
        for i in range(self._max_fill):
            items = next(self._iterator)

            if not isinstance(items, tuple):
                items = (items,)

            max_length = max([len(item) for item in items])

            if max_length < self._min_length:
                continue

            if max_length > self._max_length:
                continue

            if max_length in self._bucket_map:
                idx = self._bucket_map[max_length]
                self._buckets[idx].append(items)
            else:
                self._buckets[-1].append(items)

    def _get_content(self, idx: int) -> List:
        idx = self._priority.pop(idx)
        self._priority.append(idx)

        bucket = self._buckets[idx]
        outs = bucket[:self._batch_sizes[idx]]
        self._buckets[idx] = bucket[self._batch_sizes[idx]:]

        content = tuple([list(item) for item in zip(*outs)])
        content = self._pad_batch(content)

        if self._spec.elem_type is List[List[int]]:
            return self._pad_batch(content)[0]
        else:
            return self._pad_batch(content)

    def _pad_batch(self, batch: tuple) -> List:
        for bat in batch:
            max_len = max([len(item) for item in bat])

            for seq in bat:
                for _ in range(len(seq), max_len):
                    seq.append(self._pad)

        return batch

    def _get_bucket(self) -> int:
        for i, idx in enumerate(self._priority):
            if len(self._buckets[idx]) >= self._batch_sizes[idx]:
                return i

        return -1

    def _get_nonempty_bucket(self) -> int:
        for i, idx in enumerate(self._priority):
            if len(self._buckets[idx]) > 0:
                return i

        return -1


class _LookupDSIter(IteratorBase):

    def __init__(self, dataset: "LookupDataset"):
        self._unk_id = dataset.unk_id
        self._vocabulary = dataset.vocabulary
        self._iterator = iter(dataset._dataset)

    @_profile("_LookupDSIter", False)
    def __next__(self) -> List[int]:
        outputs = []

        for s in next(self._iterator):
            if s not in self._vocabulary:
                outputs.append(self._unk_id)
            else:
                outputs.append(self._vocabulary[s])

        return outputs


class _MapDSIter(IteratorBase):

    def __init__(self, dataset: "MapDataset"):
        self._fn = dataset._fn
        self._iterator = iter(dataset._dataset)

    @_profile("_LookupDSIter", False)
    def __next__(self) -> Any:
        item = next(self._iterator)

        return self._fn(item)


class _PaddedBatchDSIter(IteratorBase):

    def __init__(self, dataset: "PaddedBatchDataset"):
        self._pad = dataset.pad
        self._batch_size = dataset.batch_size
        self._iterator = iter(dataset._dataset)
        self._spec = dataset.element_spec

        super(_PaddedBatchDSIter, self).__init__()


    def __iter__(self) -> "_PaddedBatchDSIter":
        return self

    @_profile("_PaddedBatchDSIter", False)
    def __next__(self) -> Union[List[List[int]],
                                Tuple[List[List[int]], ...]]:
        bucket = []

        try:
            for _ in range(self._batch_size):
                bucket.append(_maybe_to_tuple(next(self._iterator)))
        except StopIteration:
            if len(bucket) == 0:
                raise StopIteration

        # unzip
        bucket = list(map(lambda x: list(x), _unzip(bucket)))
        max_lens = map(lambda x: max(list(map(lambda v: len(v), x))), bucket)

        outputs = []

        for seqs, max_len in zip(bucket, max_lens):
            outputs.append(self._pad_batch(seqs, max_len))

        if self._spec.elem_type is List[List[int]]:
            return bucket[0]
        else:
            return bucket

    def _pad_batch(self, seqs: List, max_len: int) -> List:
        for seq in seqs:
            for _ in range(max_len - len(seq)):
                seq.append(self._pad)

        return seqs

class _RepeatDSIter(IteratorBase):

    def __init__(self, dataset: "RepeatDataset"):
        self._dataset = dataset
        self._iterator = iter(dataset._dataset)
        self._n = 0
        self._count = dataset.count

    @_profile("_RepeatDSIter", False)
    def __next__(self) -> Any:
        try:
            return next(self._iterator)
        except StopIteration:
            self._n = self._n + 1

            if self._count <= 0 or self._n < self._count:
                self._iterator = iter(self._dataset)
                return next(self._iterator)

            raise StopIteration


class _ShardDSIter(IteratorBase):

    def __init__(self, dataset: "ShardDataset"):
        self._num_shards = dataset.num_shards
        self._index = dataset._index
        self._n = 0
        self._iterator = iter(dataset._dataset)

    @_profile("_ShardDsIter", False)
    def __next__(self) -> Any:
        while self._n != self._index:
            next(self._iterator)
            self._n = (self._n + 1) % self._num_shards

        self._n = (self._n + 1) % self._num_shards

        return next(self._iterator)


class _TextLineDSIter(IteratorBase):

    def __init__(self, dataset: "TextLineDataset"):
        if isinstance(dataset.input_source, str):
            self._file = open(dataset.input_source, "rb")
        else:
            self._file = _FileWrapper(dataset.input_source)

    @_profile("_TextLineDSIter", False)
    def __next__(self) -> bytes:
        return next(self._file)


class _TokenizedLineDSIter(IteratorBase):

    def __init__(self, dataset: "Dataset"):
        self._bos = dataset.bos
        self._eos = dataset.eos
        self._tokenizer = dataset.tokenizer
        self._iterator = iter(dataset._dataset)

    @_profile("_TokenizedLineDSIter", False)
    def __next__(self) -> List[bytes]:
        val = self._tokenizer.encode(next(self._iterator))

        if self._bos:
            val.insert(0, self._bos)

        if self._eos:
            val.append(self._eos)

        return val


class _ZipDSIter(IteratorBase):

    def __init__(self, dataset: "ZipDataset"):
        self._iterators = [iter(ds) for ds in dataset._datasets]

    @_profile("_ZipDSIter", False)
    def __next__(self) -> Tuple:
        outputs = []

        for iterator in self._iterators:
            outputs.append(next(iterator))

        return tuple(outputs)


_DATASET_TO_ITER = {
    "BackgroundDataset": _BackgroundDSIter,
    "BucketDataset": _BucketDSIter,
    "LookupDataset": _LookupDSIter,
    "MapDataset": _MapDSIter,
    "PaddedBatchDataset": _PaddedBatchDSIter,
    "RepeatDataset": _RepeatDSIter,
    "ShardDataset": _ShardDSIter,
    "TextLineDataset": _TextLineDSIter,
    "TokenizedLineDataset": _TokenizedLineDSIter,
    "ZipDataset": _ZipDSIter
}


class Iterator(IteratorBase):

    def __init__(self, dataset: "Dataset"):
        self._iterator = _DATASET_TO_ITER[dataset.name](dataset)

    def __next__(self):
        return next(self._iterator)
