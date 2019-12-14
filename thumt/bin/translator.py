# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import six
import time
import torch

import thumt.data as data
import thumt.models as models
import thumt.utils as utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision for decoding")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        device=0,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        decode_batch_size=32,
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
    params.parse(args.parameters)

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


def convert_to_string(tensor, params):
    tensor = torch.squeeze(tensor, dim=1)
    tensor = tensor.tolist()
    decoded = []

    for ids in tensor:
        output = []
        for wid in ids:
            if wid == 1:
                break
            output.append(params.mapping["target"][wid])
        decoded.append(b" ".join(output))

    return decoded


def main(args):
    # Load configs
    model_cls = models.get_model(args.model)
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = import_params(args.checkpoint, args.model, params)
    params = override_params(params, args)
    torch.cuda.set_device(params.device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Create model
    with torch.no_grad():
        model = model_cls(params).cuda()

        if args.half:
            model = model.half()
            torch.set_default_dtype(torch.half)
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        model.eval()
        model.load_state_dict(
            torch.load(utils.latest_checkpoint(args.checkpoint),
                       map_location="cpu")["model"])

        # Decoding
        dataset = data.get_dataset(args.input, "infer", params)
        fd = open(args.output, "wb")
        counter = 0

        for features in dataset:
            t = time.time()
            counter += 1
            features = data.lookup(features, "infer", params)

            seqs, _ = utils.beam_search([model], features, params)
            batch = convert_to_string(seqs, params)

            for seq in batch:
                fd.write(seq)
                fd.write(b"\n")

            t = time.time() - t
            print("Finished batch: %d (%.3f sec)" % (counter, t))

        fd.close()


if __name__ == "__main__":
    main(parse_args())
