# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import six

import tensorflow as tf


def load_vocab(filename):
    with tf.gfile.Open(filename) as fd:
        count = 0
        vocab = {}
        for line in fd:
            word = line.strip()
            vocab[word] = count
            count += 1

    return vocab


def to_example(dictionary):
    """ Convert python dictionary to tf.train.Example """
    features = {}

    for (k, v) in six.iteritems(dictionary):
        if not v:
            raise ValueError("Empty generated field: %s", str((k, v)))

        if isinstance(v[0], six.integer_types):
            int64_list = tf.train.Int64List(value=v)
            features[k] = tf.train.Feature(int64_list=int64_list)
        elif isinstance(v[0], float):
            float_list = tf.train.FloatList(value=v)
            features[k] = tf.train.Feature(float_list=float_list)
        elif isinstance(v[0], six.string_types):
            bytes_list = tf.train.BytesList(value=v)
            features[k] = tf.train.Feature(bytes_list=bytes_list)
        else:
            raise ValueError("Value is neither an int nor a float; "
                             "v: %s type: %s" % (str(v[0]), str(type(v[0]))))

    return tf.train.Example(features=tf.train.Features(feature=features))


def write_records(records, out_filename):
    """ Write to TensorFlow record """
    writer = tf.python_io.TFRecordWriter(out_filename)

    for count, record in enumerate(records):
        writer.write(record)
        if count % 10000 == 0:
            tf.logging.info("write: %d", count)

    writer.close()


def convert_to_record(inputs, vocab, output_name, output_dir, num_shards,
                      shuffle=False):
    """ Convert plain parallel text to TensorFlow record """
    source, target = inputs
    svocab, tvocab = vocab
    records = []

    with tf.gfile.Open(source) as src:
        with tf.gfile.Open(target) as tgt:
            for sline, tline in zip(src, tgt):
                sline = sline.strip().split()
                sline = [svocab[item] if item in svocab else svocab[FLAGS.unk]
                         for item in sline] + [svocab[FLAGS.eos]]
                tline = tline.strip().split()
                tline = [tvocab[item] if item in tvocab else tvocab[FLAGS.unk]
                         for item in tline] + [tvocab[FLAGS.eos]]

                feature = {
                    "source": sline,
                    "target": tline,
                    "source_length": [len(sline)],
                    "target_length": [len(tline)]
                }
                records.append(feature)

    output_files = []
    writers = []

    for shard in xrange(num_shards):
        output_filename = "%s-%.5d-of-%.5d" % (output_name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        output_files.append(output_file)
        writers.append(tf.python_io.TFRecordWriter(output_file))

    counter, shard = 0, 0

    if shuffle:
        random.shuffle(records)

    for record in records:
        counter += 1
        example = to_example(record)
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % num_shards

    for writer in writers:
        writer.close()


def parse_args():
    msg = "convert inputs to tf.Record format"
    usage = "input_converter.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    parser.add_argument("--input", required=True, type=str, nargs=2,
                        help="Path of input file")
    parser.add_argument("--output_name", required=True, type=str,
                        help="Output name")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory")
    parser.add_argument("--vocab", nargs=2, required=True, type=str,
                        help="Path of vocabulary")
    parser.add_argument("--num_shards", default=100, type=int,
                        help="Number of output shards")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle inputs")
    parser.add_argument("--unk", default="<unk>", type=str,
                        help="Unknown word symbol")
    parser.add_argument("--eos", default="<eos>", type=str,
                        help="End of sentence symbol")

    return parser.parse_args()


def main(_):
    svocab = load_vocab(FLAGS.vocab[0])
    tvocab = load_vocab(FLAGS.vocab[1])

    # convert data
    convert_to_record(FLAGS.input, [svocab, tvocab], FLAGS.output_name,
                      FLAGS.output_dir, FLAGS.num_shards, FLAGS.shuffle)


if __name__ == "__main__":
    FLAGS = parse_args()
    tf.app.run()
