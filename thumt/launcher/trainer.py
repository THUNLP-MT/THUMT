#!/usr/bin/env python
# coding=utf-8
# Copyright 2017 The THUMT Authors

import os
import argparse
import numpy as np
import tensorflow as tf

import thumt.models as models
import thumt.utils.search as search
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary

from thumt.utils.parallel import parallel_model
from thumt.utils.hooks import BLEUHook, get_or_create_bleu_tensor


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path of source and target corpus")
    parser.add_argument("--output", type=str, help="Path to saved models")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, nargs="+",
                        help="Path of reference files")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        model=None,
        # Default training hyper parameters
        num_threads=6,
        batch_size=128,
        max_length=256,
        length_multiplier=1,
        mantissa_bits=1,
        warmup_steps=4000,
        train_steps=100000,
        buffer_size=10000,
        constant_batch_size=True,
        device_list=[0],
        initializer="uniform",
        initializer_gain=0.08,
        learning_rate=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        clip_grad_norm=5.0,
        learning_rate_decay="noam",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        keep_checkpoint_max=20,
        # Validation
        eval_steps=2000,
        eval_batch_size=32,
        alpha=0.6,
        top_beams=1,
        beam_size=4,
        decode_length=50,
        validation="",
        references=[""],
        save_checkpoint_secs=600
    )

    return params


def import_params(model_dir, params):
    p_name = os.path.join(model_dir, "/params.json")
    m_name = os.path.join(model_dir, params.model + ".json")
    with tf.gfile.Open(p_name) as fd:
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in params.values().iterkeys():
        collected.add_hparam(k, getattr(all_params, k))

    return collected


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


def override_parameters(params, args):
    params.input = args.input
    params.output = args.output
    params.model = args.model
    params.parse(args.parameters)

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
    params.validation = args.validation
    params.references = args.references

    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay == "noam":
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def decode_target_ids(inputs, params):
    decoded = []
    vocab = params.vocabulary["target"]

    for item in inputs:
        syms = []
        for idx in item:
            sym = vocab[idx]

            if sym == params.eos:
                break

            if sym == params.pad:
                break

            syms.append(sym)
        decoded.append(syms)

    return decoded


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()
    params = merge_parameters(params, model_cls.get_parameters())
    params = override_parameters(params, args)

    # Import parameters

    # Export all parameters and model specific parameters
    export_params(params.output, "params.json", params)
    export_params(
        params.output,
        "%s.json" % args.model,
        collect_params(params, model_cls.get_parameters())
    )

    # Build Graph
    with tf.Graph().as_default():
        # Build input queue
        features = dataset.get_training_input(params.input, params)

        # Build model
        initializer = get_initializer(params)
        model = model_cls(params)

        # Multi-GPU setting
        sharded_losses = parallel_model(
            model.get_training_func(initializer),
            features,
            params.device_list
        )
        loss = tf.add_n(sharded_losses) / len(sharded_losses)

        # Create global step
        global_step = tf.train.get_or_create_global_step()

        # Print parameters
        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0

        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
            v_size = np.prod(np.array(v.shape.as_list())).tolist()
            total_size += v_size
        tf.logging.info("Total trainable variables size: %d", total_size)

        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        tf.summary.scalar("learning_rate", learning_rate)

        # Create optimizer
        opt = tf.train.AdamOptimizer(learning_rate,
                                     beta1=params.adam_beta1,
                                     beta2=params.adam_beta2,
                                     epsilon=params.adam_epsilon)

        train_op = tf.contrib.layers.optimize_loss(
            name="training",
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            clip_gradients=params.clip_grad_norm or None,
            optimizer=opt,
            colocate_gradients_with_ops=True
        )

        # Validation
        if params.validation and params.references[0]:
            files = [params.validation] + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            eval_input_fn = dataset.get_evaluation_input
        else:
            eval_input_fn = None

        # Validation
        eval_fn = lambda f: search.create_inference_graph(
            model.get_evaluation_func(), f, params
        )

        # Add hooks
        hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "loss": loss,
                    "source": tf.shape(features["source"]),
                    "target": tf.shape(features["target"])
                },
                every_n_iter=1
            )
        ]

        config = session_config(params)

        if eval_input_fn is not None:
            score_tensor = get_or_create_bleu_tensor()
            hooks.append(
                BLEUHook(
                    score_tensor,
                    eval_fn,
                    lambda: eval_input_fn(eval_inputs, params),
                    lambda x: decode_target_ids(x, params),
                    params.output,
                    config,
                    os.path.join(params.output, "best"),
                    eval_steps=params.eval_steps,
                    saver=tf.train.Saver(
                        save_relative_paths=True
                    )
                )
            )

        # scaffold -> MonitoredTrainingSession -> CheckpointSaverHook
        scaffold = tf.train.Scaffold(
            saver=tf.train.Saver(
                max_to_keep=params.keep_checkpoint_max
            )
        )

        # Create session
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=hooks,
                save_checkpoint_secs=params.save_checkpoint_secs,
                scaffold=scaffold, config=config) as sess:
            while not sess.should_stop():
                sess.run(train_op)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parse_args())
