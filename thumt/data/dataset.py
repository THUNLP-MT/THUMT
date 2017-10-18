# coding=utf-8
# Copyright 2017 The THUMT Authors

import math
import operator
import numpy as np
import tensorflow as tf


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on maximum length
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs


def get_training_input(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0)
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt: {
                "source": src,
                "target": tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        example = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )

        # String to index lookup
        example["source"] = src_table.lookup(example["source"])
        example["target"] = tgt_table.lookup(example["target"])

        # Batching
        examples = batch_examples(example, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        examples["source"] = tf.to_int32(examples["source"])
        examples["target"] = tf.to_int32(examples["target"])
        examples["source_length"] = tf.to_int32(examples["source_length"])
        examples["target_length"] = tf.to_int32(examples["target_length"])
        examples["source_length"] = tf.squeeze(examples["source_length"], 1)
        examples["target_length"] = tf.squeeze(examples["target_length"], 1)

        return examples


def sort_input_file(filename, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs


def get_inference_input(inputs, params):
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(inputs)
    )

    # Split string
    dataset = dataset.map(lambda x: tf.string_split([x]).values,
                          num_parallel_calls=params.num_threads)

    # Append <eos>
    dataset = dataset.map(
        lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
        num_parallel_calls=params.num_threads
    )

    # Convert tuple to dictionary
    dataset = dataset.map(
        lambda x: {"source": x, "source_length": tf.shape(x)[0]},
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.padded_batch(
        params.decode_batch_size,
        {"source": [tf.Dimension(None)], "source_length": []},
        {"source": params.pad, "source_length": 0}
    )

    iterator = dataset.make_one_shot_iterator()
    examples = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["source"]),
        default_value=params.mapping["source"][params.unk]
    )
    examples["source"] = src_table.lookup(examples["source"])

    return examples
