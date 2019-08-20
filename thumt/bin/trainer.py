# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import glob
import itertools
import logging
import os
import six
import socket
import time
import torch

import torch.distributed as dist
import thumt.data as data
import thumt.utils as utils
import thumt.models as models
import thumt.modules as modules
import thumt.optimizers as optimizers


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path of source and target corpus")
    parser.add_argument("--record", type=str,
                        help="Path to tf.Record data")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, nargs="+",
                        help="Path of reference files")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training mode")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank of this process")
    parser.add_argument("--half", action="store_true",
                        help="Enable mixed precision training")
    parser.add_argument("--hparam_set", type=str,
                        help="Name of pre-defined hyper parameter set")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")

    return parser.parse_args(args)


def default_params():
    params = utils.HParams(
        input=["", ""],
        output="",
        model="transformer",
        vocab=["", ""],
        pad="<pad>",
        bos="<eos>",
        eos="<eos>",
        unk="<unk>",
        # Dataset
        batch_size=4096,
        batch_multiplier=1,
        fixed_batch_size=False,
        min_length=1,
        max_length=256,
        buffer_size=10000,
        # Initialization
        initializer_gain=1.0,
        initializer="uniform_unit_scaling",
        # Regularization
        scale_l1=0.0,
        scale_l2=0.0,
        # Training
        warmup_steps=4000,
        train_steps=100000,
        update_cycle=1,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        clip_grad_norm=5.0,
        learning_rate=1.0,
        learning_rate_schedule="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        device_list=[0],
        # Checkpoint Saving
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
        # Validation
        eval_steps=2000,
        eval_secs=0,
        eval_batch_size=32,
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        validation="",
        references=[""],
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(p_name) or not os.path.exists(m_name):
        return params

    with open(p_name) as fd:
        logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)

    with open(filename, "w") as fd:
        fd.write(params.to_json())


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_params(params, args):
    params.model = args.model or params.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters)

    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(params.vocab[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(params.vocab[1])

    params.vocabulary = {
        "source": src_vocab, "target": tgt_vocab
    }
    params.lookup = {
        "source": src_w2idx, "target": tgt_w2idx
    }
    params.mapping = {
        "source": src_idx2w, "target": tgt_idx2w
    }

    return params


def collect_params(all_params, params):
    collected = utils.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def print_variables(model):
    weights = {v[0]: v[1] for v in model.named_parameters()}
    total_size = 0

    for name in sorted(list(weights)):
        v = weights[name]
        print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))
        total_size += v.nelement()

    print("Total trainable variables size: %d" % total_size)


def save_checkpoint(step, epoch, model, optimizer, params):
    if dist.get_rank() == 0:
        state = {"step": step, "epoch": epoch, "model": model.state_dict()}
        utils.save(state, params.output, params.keep_checkpoint_max)


def infer_gpu_num(s):
    kv_list = s.split(",")
    kv_list = [kv.split("=") for kv in kv_list]
    kv_dict = {item[0]: item[1] for item in kv_list}

    if "device_list" not in kv_dict:
        return 1
    else:
        dev_str = kv_dict["device_list"].lstrip("[").rstrip("]")
        return len(dev_str.split(","))


def broadcast(model, optimizer):
    for var in model.parameters():
        dist.broadcast(var.data, 0)


def main(args):
    model_cls = models.get_model(args.model)

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = default_params()
    params = merge_params(params, model_cls.default_params(args.hparam_set))
    params = import_params(args.output, args.model, params)
    params = override_params(params, args)

    # Initialize distributed utility
    if args.distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(args.local_rank)
    else:
        dist.init_process_group("nccl", init_method=args.url,
                                rank=args.local_rank,
                                world_size=len(params.device_list))
        torch.cuda.set_device(params.device_list[args.local_rank])

    # Export parameters
    if dist.get_rank() == 0:
        export_params(params.output, "params.json", params)
        export_params(params.output, "%s.json" % params.model,
                      collect_params(params, model_cls.default_params()))

    model = model_cls(params).cuda()

    if args.half:
        model = model.half()

    model.train()
    criterion = modules.SmoothedCrossEntropyLoss(params.label_smoothing)
    schedule = optimizers.LinearWarmupRsqrtDecay(params.learning_rate,
                                                 params.warmup_steps)
    optimizer = optimizers.AdamOptimizer(learning_rate=schedule,
                                         beta_1=params.adam_beta1,
                                         beta_2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
    optimizer = optimizers.MultiStepOptimizer(optimizer, params.update_cycle)

    if args.half:
        optimizer = optimizers.LossScalingOptimizer(optimizer)

    if dist.get_rank() == 0:
        print_variables(model)

    dataset = data.get_dataset(params.input, "train", params)

    # Load checkpoint
    checkpoint = utils.latest_checkpoint(params.output)

    if checkpoint is not None:
        state = torch.load(checkpoint)
        step = state["step"]
        epoch = state["epoch"]
        model.load_state_dict(state["model"])
    else:
        step = 0
        epoch = 0
        broadcast(model, optimizer)

    def train_fn(features):
        labels = features["labels"]
        logits = model(features)
        loss = criterion(logits, labels)
        mask = torch.ne(labels, 0).to(loss)
        return torch.sum(loss * mask) / torch.sum(mask)

    counter = 0
    should_save = False

    while True:
        for features in dataset:
            if counter % params.update_cycle == 0:
                step += 1
                should_save = True

            counter += 1
            t = time.time()
            features = data.lookup(features, "train", params)
            loss = train_fn(features)
            gradients = optimizer.compute_gradients(loss,
                                                    model.parameters())
            optimizer.apply_gradients(zip(gradients,
                                          list(model.parameters())))

            t = time.time() - t

            print("epoch = %d, step = %d, loss = %.3f (%.3f sec)" %
                  (epoch + 1, step, float(loss), t))

            if step % params.save_checkpoint_steps == 0:
                if should_save:
                    save_checkpoint(step, epoch, model, optimizer, params)
                    should_save = False

            if step >= params.train_steps:
                if should_save:
                    save_checkpoint(step, epoch, model, optimizer, params)
                return

        epoch += 1


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


if __name__ == "__main__":
    parsed_args = parse_args()

    if parsed_args.distributed:
        main(parsed_args)
    else:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            parsed_args.url = url

        world_size = infer_gpu_num(parsed_args.parameters)
        torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                    nprocs=world_size)
