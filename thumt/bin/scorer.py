#! /usr/bin python
# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import six
import time
import copy
import torch
import socket
import logging
import argparse
import numpy as np

import torch.distributed as dist

import thumt.data as data
import thumt.utils as utils
import thumt.models as models

logging.getLogger().setLevel(logging.INFO)
# just show error messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score input sentences with pre-trained checkpoints.",
        usage="scorer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True, nargs=2,
                        help="Path to input file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint.")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path to source and target vocabulary.")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper-parameters.")
    parser.add_argument("--half", action="store_true",
                        help="Enable Half-precision for decoding.")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        vocabulary=None,
        model=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        append_eos=False,
        monte_carlo=False,
        device_list=[0],
        decode_batch_size=32,
        buffer_size=10000,
        level="sentence"
    )

    return params


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


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    if args.parameters:
        params.parse(args.parameters.lower())

    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(args.vocabulary[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(args.vocabulary[1])

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


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1

    dev_str = result.groups()[-1]
    return len(dev_str.split(","))


def main(args):
    model_cls = models.get_model(args.model)
    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = import_params(args.checkpoint, args.model, params)
    params = override_params(params, args)

    dist.init_process_group("nccl", init_method=args.url,
                            rank=args.local_rank,
                            world_size=len(params.device_list))
    torch.cuda.set_device(params.device_list[args.local_rank])
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    def score_fn(inputs, _model, level="sentence"):
        _features, _labels = inputs
        _score = _model(_features, _labels, mode="eval", level=level)
        return _score

    # Create model
    with torch.no_grad():
        model = model_cls(params).cuda()

        if args.half:
            model = model.half()

        if not params.monte_carlo:
            model.eval()

        model.load_state_dict(
            torch.load(utils.latest_checkpoint(args.checkpoint),
                       map_location="cpu")["model"])
        dataset = data.get_dataset(args.input, "eval", params)
        data_iter = iter(dataset)
        counter = 0
        pad_max = 1024

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        if params.level == "sentence":
            t_list = [torch.empty([params.decode_batch_size]).float()
                      for _ in range(dist.get_world_size())]
        else:
            t_list = [torch.empty([params.decode_batch_size, pad_max]).float()
                      for _ in range(dist.get_world_size())]

        if dist.get_rank() == 0:
            fd = open(args.output, "w")
        else:
            fd = None

        while True:
            try:
                features = next(data_iter)
                features = data.lookup(features, "eval", params)
                batch_size = features[0]["source"].shape[0]
            except:
                features = {
                    "source": torch.ones([1, 1]).long(),
                    "source_mask": torch.ones([1, 1]).float(),
                    "target": torch.ones([1, 1]).long(),
                    "target_mask": torch.ones([1, 1]).float()
                }, torch.ones([1, 1]).long()
                batch_size = 0

            t = time.time()
            counter += 1

            scores = score_fn(features, model, params.level)

            # Padding
            if params.level == "sentence":
                pad_batch = params.decode_batch_size - scores.shape[0]
                scores = torch.nn.functional.pad(scores, [0, pad_batch])
            else:
                pad_batch = params.decode_batch_size - scores.shape[0]
                pad_length = pad_max - scores.shape[1]
                scores = torch.nn.functional.pad(
                    scores, (0, pad_length, 0, pad_batch), value=-1)

            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, scores.float())

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            for i in range(params.decode_batch_size):
                for j in range(dist.get_world_size()):
                    n = size[j]
                    score = t_list[j][i]

                    if i >= n:
                        continue

                    if params.level == "sentence":
                        fd.write("{:.4f}\n".format(score))
                    else:
                        s_list = score.tolist()
                        for s in s_list:
                            if s >= 0:
                                fd.write("{:.8f} ".format(s))
                            else:
                                fd.write("\n")
                                break

            t = time.time() - t
            logging.info("Finished batch: %d (%.3f sec)" % (counter, t))

        if dist.get_rank() == 0:
            fd.close()


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    # Pick a free port
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        parsed_args.url = url

    world_size = infer_gpu_num(parsed_args.parameters)

    if world_size > 1:
        torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                    nprocs=world_size)
    else:
        process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
