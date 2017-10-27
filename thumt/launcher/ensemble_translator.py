#!/usr/bin/env python
# coding=utf-8
# Copyright 2017 The THUMT Authors

import argparse
import itertools
import numpy as np
import tensorflow as tf

import thumt.models as models
import thumt.utils.search as search
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using neural machine translation models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, nargs="+",
                        help="Additional hyper parameters")

    return parser.parse_args()


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


def override_parameters(params, args, i):
    params.input = args.input
    params.output = args.output

    if args.parameters:
        params.parse(args.parameters[i])

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary[1])
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


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var_name.find("feed_forward") >= 0:
                var_name = var_name.replace("feed_forward", "computation")

            if var.name[:-2] == var_name:
                print("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break

    return ops


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [models.get_model(model) for model in args.models]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.model_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        override_parameters(params_list[i], args, i)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for checkpoint in args.checkpoints:
            print("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                # TODO: more general cases
                if not name.startswith("rnnsearch"):
                    continue

                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor

            model_var_lists.append(values)

        # Build models
        model_fns = []

        for i in range(len(args.checkpoints)):
            # TODO: replace rnnsearch with model_cls.name
            model = model_cls_list[i](params_list[i], "rnnsearch_%d" % i)
            model_fn = model.get_inference_fn()
            model_fns.append(model_fn)

        # Read input file
        sorted_keys, sorted_inputs = dataset.sort_input_file(args.input)
        # Build input queue
        features = dataset.get_inference_input(sorted_inputs, params_list[0])
        predictions = search.create_inference_graph(model_fns, features,
                                                    params_list)

        assign_ops = []

        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []

            for v in all_var_list:
                if v.name.startswith("rnnsearch_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                               "rnnsearch_%d" % i)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)

        session_creator = tf.train.ChiefSessionCreator(
            config=session_config(params_list[0])
        )

        results = []

        # Create session
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            sess.run(assign_op)

            while not sess.should_stop():
                results.append(sess.run(predictions))
                tf.logging.log(tf.logging.INFO, "Finished batch %d" % len(results))

    # Convert to plain text
    vocab = params_list[0].vocabulary["target"]
    outputs = []

    for result in results:
        outputs.append(result.tolist())

    outputs = list(itertools.chain(*outputs))

    restored_outputs = []

    for index in range(len(sorted_inputs)):
        restored_outputs.append(outputs[sorted_keys[index]])

    # Write to file
    with open(args.output, "w") as outfile:
        for output in restored_outputs:
            decoded = [vocab[idx] for idx in output]
            decoded = " ".join(decoded)
            idx = decoded.find(params_list[0].eos)

            if idx >= 0:
                output = decoded[:idx].strip()
            else:
                output = decoded.strip()

            outfile.write("%s\n" % output)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
