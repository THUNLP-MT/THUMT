#!/usr/bin/env python
# coding=utf-8
# Copyright 2017 The THUMT Authors

import argparse
import itertools
import numpy as np
import tensorflow as tf

import thumt.models as models
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Path to input file")
flags.DEFINE_string("output", "", "Path to output file")
flags.DEFINE_string("path", "thumt_train", "Path to checkpoints")
flags.DEFINE_string("model", "rnnsearch", "Name of the model to train")
flags.DEFINE_string("parameters", "", "Optional parameters")


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        model=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        # decoding
        alpha=0.6,
        top_beams=1,
        beam_size=4,
        decode_length=50,
        decode_batch_size=32,
        device_list=[0],
        num_threads=6
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().iteritems():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().iteritems():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def override_parameters(params):
    params.input = FLAGS.input
    params.output = FLAGS.output
    params.path = FLAGS.path
    params.parse(FLAGS.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(params.vocabulary[0]),
        "target": vocabulary.load_vocabulary(params.vocabulary[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )
    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            [params.pad, params.bos, params.eos, params.unk]
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            [params.pad, params.bos, params.eos, params.unk]
        )
    }

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()
    params = merge_parameters(params, model_cls.model_parameters())

    params = override_parameters(params)

    # Build Graph
    with tf.Graph().as_default():
        # Read input file
        sorted_keys, sorted_inputs = dataset.sort_input_file(params.input)
        # Build input queue
        features = dataset.get_inference_input(sorted_inputs, params)

        # Build model
        model = model_cls(params)
        predictions = model.build_inference_graph(features)

        session_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=params.path,
            config=session_config(params)
        )

        results = []

        with tf.train.MonitoredSession(
                session_creator=session_creator) as sess:
            while not sess.should_stop():
                results.append(sess.run(predictions))
                tf.logging.log(tf.logging.INFO,
                               "Finished batch %d" % len(results))

    # Convert to plain text
    vocab = params.vocabulary["target"]
    outputs = []

    for result in results:
        outputs.append(result.tolist())

    outputs = list(itertools.chain(*outputs))

    restored_outputs = []

    for index in range(len(sorted_inputs)):
        restored_outputs.append(outputs[sorted_keys[index]])

    # Write to file
    with open(params.output, "w") as outfile:
        for output in restored_outputs:
            decoded = [vocab[idx] for idx in output]
            decoded = " ".join(decoded)
            idx = decoded.find(params.eos)

            if idx >= 0:
                output = decoded[:idx].strip()
            else:
                output = decoded.strip()

            outfile.write("%s\n" % output)


if __name__ == "__main__":
    tf.app.run()
