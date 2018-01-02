# coding=utf-8
# Code modified from Tensor2Tensor library
# Copyright 2018 The THUMT Authors

import tensorflow as tf
import numpy
import json
import math
import thumt.utils.bleu as bleu

# Default value for INF
INF = 1. * 1e7


def get_mrt_features(features, params, model):
    # Generate samples
    samples = create_sampling_graph(model.get_inference_func(), features,
                                    params, training=True)

    eos_id = params.mapping["target"][params.eos]
    features["samples"] = samples
    # Delete bos & add eos
    features["samples"] = features["samples"][:, 1:]
    sample_shape = tf.shape(features["samples"])
    eos_seq = tf.ones([sample_shape[0], 1]) * eos_id
    eos_seq = tf.to_int32(eos_seq)
    features["samples"] = tf.concat([features["samples"], eos_seq], 1)

    # Add the gold reference
    pad_num = (tf.shape(features["samples"])[1] -
               tf.shape(features["target"])[1])
    padding = tf.zeros((1, pad_num), dtype=tf.int32)
    target_pad = tf.concat([features["target"], padding], axis=1)
    features["samples"] = tf.concat([features["samples"], target_pad],
                                    axis=0)
    # Delete repetition
    features["samples"] = tf.py_func(get_unique, [features["samples"], eos_id],
                                     tf.int32)
    features["samples"].set_shape([None, None])
    sample_shape = tf.shape(features["samples"])
    # Get sentence length
    features["sample_length"] = get_len(features["samples"], eos_id)
    # Transform to int32
    features["samples"] = tf.to_int32(features["samples"])
    features["sample_length"] = tf.to_int32(features["sample_length"])
    # Repeat source sentences
    features["source"] = tf.tile(features["source"], [sample_shape[0], 1])
    features["source_length"] = tf.tile(features["source_length"],
                                        [sample_shape[0]])
    # Calculate BLEU
    bleu_fn = lambda x: bleu_tensor(x, features["target"], eos_id)
    features["BLEU"] = tf.map_fn(bleu_fn, features["samples"],
                                 dtype=tf.float32)
    features["BLEU"].set_shape((None,))
    # Set target
    features["target"] = features["samples"]
    features["target_length"] = features["sample_length"]
    return features


def cut_sen(sen, eos):
    if not eos in sen:
        return sen
    else:
        pos_eos = sen.index(eos)
        return sen[:pos_eos+1]


def get_unique(sens, eos):
    sens = sens.tolist()
    result = []
    maxlen = -1
    # remove repetition
    for sen in sens:
        tmp = cut_sen(sen, eos)
        if tmp not in result:
            result.append(tmp)
            if len(tmp) > maxlen:
                maxlen = len(tmp)
    result = [sen + [eos] * (maxlen - len(sen)) for sen in result]
    result = numpy.asarray(result, dtype=numpy.int32)
    return result


def get_len(sen, eos):
    indices = tf.where(tf.equal(sen, eos))
    result = tf.segment_min(indices[:,1], indices[:,0])
    return result


def log_prob_from_logits(logits):
    return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)


def sampler(symbols_to_logits_fn, initial_ids, sample_num, decode_length,
            vocab_size, eos_id, features=None):
    batch_size = tf.shape(initial_ids)[0]

    # Expand each batch to sample_num
    seqlen = tf.constant(0)
    alive_seq = tf.tile(tf.expand_dims(initial_ids, 1), [1, sample_num])
    alive_seq = tf.expand_dims(alive_seq, 2)  # (batch_size, sample_num, 1)
    sa = tf.shape(alive_seq)
    alive_seq = tf.reshape(alive_seq, [sa[0]*sa[1],1])

    def _is_finished(i, alive_seq):
        return i < decode_length

    def inner_loop(i, alive_seq):
        logit = symbols_to_logits_fn(alive_seq)[0]
        new_samples = tf.multinomial(logit, 1)
        new_samples = tf.to_int32(new_samples)
        alive_seq = tf.concat([alive_seq, new_samples], 1)
        return (i + 1, alive_seq)

    (_, alive_seq) = tf.while_loop(
        _is_finished,
        inner_loop,
        [seqlen, alive_seq],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, None])
        ],
        parallel_iterations=1,
        back_prop=False
    )
    alive_seq.set_shape((sample_num, None))

    return alive_seq


def create_sampling_graph(model_fns, features, params, training = False):
    if isinstance(params, (list, tuple)):
        params_list = params
        params = params_list[0]
    else:
        params_list = [params]

    if not isinstance(model_fns, (list, tuple)):
        model_fns = [model_fns]

    decode_length = params.decode_length
    sample_num = params.mrt_sample
    top_beams = params.top_beams

    # [batch, decoded_ids] => [batch, vocab_size]
    def symbols_to_logits_fn(decoded_ids):
        features["target"] = tf.pad(decoded_ids[:, 1:], [[0, 0], [0, 1]])
        features["target_length"] = tf.fill([tf.shape(features["target"])[0]],
                                            tf.shape(features["target"])[1])

        results = []

        for i, model_fn in enumerate(model_fns):
            results.append(model_fn(features, params_list[i]))

        return results

    batch_size = tf.shape(features["source"])[0]
    # append <bos> symbol
    bos_id = params.mapping["target"][params.bos]
    initial_ids = tf.fill([batch_size], tf.constant(bos_id, dtype=tf.int32))

    inputs_old = features["source"]
    inputs_length_old = features["source_length"]
    if training:
        outputs_old = features["target"]
        outputs_length_old = features["target_length"]

    #return
    # Expand the inputs in to the number of samples
    # [batch, length] => [batch, sample_num, length]
    features["source"] = tf.expand_dims(features["source"], 1)
    features["source"] = tf.tile(features["source"], [1, sample_num, 1])
    shape = tf.shape(features["source"])

    # [batch, sample_num, length] => [batch * sample_num, length]
    features["source"] = tf.reshape(features["source"],
                                    [shape[0] * shape[1], shape[2]])

    #return
    # For source sequence length
    features["source_length"] = tf.expand_dims(features["source_length"], 1)
    features["source_length"] = tf.tile(features["source_length"],
                                        [1, sample_num])
    shape = tf.shape(features["source_length"])

    # [batch, sample_num, length] => [batch * sample_num, length]
    features["source_length"] = tf.reshape(features["source_length"],
                                    [shape[0] * shape[1]])

    vocab_size = len(params.vocabulary["target"])
    # Setting decode length to input length + decode_length
    decode_length = tf.to_float(tf.shape(features["target"])[1]) \
                        * tf.constant(params.mrt_length_ratio)
    decode_length = tf.to_int32(decode_length)

    ids = sampler(symbols_to_logits_fn, initial_ids, params.mrt_sample,
                  decode_length, vocab_size,
                  eos_id=params.mapping["target"][params.eos],
                  features=features)

    # Set inputs back to the unexpanded inputs to not to confuse the Estimator
    features["source"] = inputs_old
    features["source_length"] = inputs_length_old
    if training:
        features["target"] = outputs_old
        features["target_length"] = outputs_length_old

    return ids


def mrt_loss(features, params, ce, tgt_mask):
    logprobs = tf.reduce_sum(ce * tgt_mask, axis=1)
    logprobs *= params.mrt_alpha
    logprobs -= tf.reduce_min(logprobs)
    probs = tf.exp(-logprobs)
    probs /= tf.reduce_sum(probs)
    ave_bleu = probs * features["BLEU"]
    loss = -tf.reduce_sum(ave_bleu)

    return loss


def bleu_tensor(trans, ref, eos):
    return tf.py_func(lambda x,y: bleu_numpy(x,y,eos,smooth=True),
                      [trans, ref], tf.float32)


def bleu_numpy(trans, refs, eos, bp="closest", smooth=False, n=4,
               weights=None):
    trans = trans.tolist()
    refs = refs.tolist()
    # cut sentence
    trans = cut_sen(trans, eos)
    refs = cut_sen(refs, eos)
    # wrap
    trans = [trans]
    refs = [refs]
    p_norm = [0 for _ in range(n)]
    p_denorm = [0 for _ in range(n)]

    for candidate, references in zip(trans, refs):
        for i in range(n):
            ccount, tcount = bleu.modified_precision(candidate, references,
                                                     i + 1)
            p_norm[i] += ccount
            p_denorm[i] += tcount

    bleu_n = [0 for _ in range(n)]

    for i in range(n):
        # add one smoothing
        if smooth and i > 0:
            p_norm[i] += 1
            p_denorm[i] += 1

        if p_norm[i] == 0 or p_denorm[i] == 0:
            bleu_n[i] = -9999
        else:
            bleu_n[i] = math.log(float(p_norm[i]) / float(p_denorm[i]))

    if weights:
        if len(weights) != n:
            raise ValueError("len(weights) != n: invalid weight number")
        log_precision = sum([bleu_n[i] * weights[i] for i in range(n)])
    else:
        log_precision = sum(bleu_n) / float(n)

    bp = bleu.brevity_penalty(trans, refs, bp)

    score = bp * math.exp(log_precision)

    return numpy.float32(score)
