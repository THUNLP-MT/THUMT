# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import tensorflow as tf


def build_input_fn(filenames, mode, params):
    def train_input_fn():
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())
        dataset = dataset.prefetch(params.buffer_size)
        dataset = dataset.shuffle(params.buffer_size)

        # Split string
        dataset = dataset.map(
            lambda x, y: (tf.strings.split([x]).values,
                          tf.strings.split([y]).values),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Append BOS and EOS
        dataset = dataset.map(
            lambda x, y: (
                (tf.concat([x, [tf.constant(params.eos)]], axis=0),
                 tf.concat([[tf.constant(params.bos)], y], axis=0)),
                tf.concat([y, [tf.constant(params.eos)]], axis=0)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        mult = params.batch_multiplier
        batch_size = params.batch_size * params.batch_multiplier
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(mult, (batch_size // (x - 1)) // mult * mult)
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        def element_length_func(x, y):
            (src, tgt), _ = x, y
            return tf.maximum(tf.shape(src)[0], tf.shape(tgt)[0])

        def valid_size(x, y):
            size = element_length_func(x, y)
            return tf.logical_and(size >= min_length, size <= max_length)

        transformation_fn = tf.data.experimental.bucket_by_sequence_length(
            element_length_func,
            boundaries,
            batch_sizes,
            padded_shapes=(
                (tf.TensorShape([None]), tf.TensorShape([None])),
                tf.TensorShape([None])),
            padding_values=(
                (tf.constant(params.pad), tf.constant(params.pad)),
                tf.constant(params.pad)),
            pad_to_bucket_boundary=True)

        dataset = dataset.filter(valid_size)
        dataset = dataset.apply(transformation_fn)

        return dataset

    def eval_input_fn():
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])
        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        # Split string
        dataset = dataset.map(
            lambda x, y: (tf.strings.split([x]).values,
                          tf.strings.split([y]).values),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Append BOS and EOS
        dataset = dataset.map(
            lambda x, y: (
                (tf.concat([x, [tf.constant(params.eos)]], axis=0),
                 tf.concat([[tf.constant(params.bos)], y], axis=0)),
                tf.concat([y, [tf.constant(params.eos)]], axis=0)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Batching
        dataset = dataset.padded_batch(
            params.batch_size,
            padded_shapes=((tf.TensorShape([None]), tf.TensorShape([None])),
                           tf.TensorShape([None])),
            padding_values=((tf.constant(params.pad), tf.constant(params.pad)),
                            tf.constant(params.pad)))

        return dataset

    def infer_input_fn():
        dataset = tf.data.TextLineDataset(filenames)

        dataset = dataset.map(
            lambda x: tf.strings.split([x]).values,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(
            params.decode_batch_size,
            padded_shapes=tf.TensorShape([None]),
            padding_values=tf.constant(params.pad))

        return dataset

    if mode == "train":
        return train_input_fn
    if mode == "eval":
        return eval_input_fn
    elif mode == "infer":
        return infer_input_fn
    else:
        raise ValueError("Unknown mode %s" % mode)


def get_dataset(filenames, mode, params):
    input_fn = build_input_fn(filenames, mode, params)

    with tf.device("/cpu:0"):
        dataset = input_fn()

    return dataset
