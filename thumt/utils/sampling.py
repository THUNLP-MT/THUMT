# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import thumt.utils.common as utils

from collections import namedtuple
from tensorflow.python.util import nest


class SamplerState(namedtuple("SamplerState",
                              ("inputs", "state", "scores", "flags"))):
    pass


def _get_inference_fn(model_fns, features):
    def inference_fn(inputs, state):
        local_features = {
            "source": features["source"],
            "source_length": features["source_length"],
            # [bos_id, ...] => [..., 0]
            "target": tf.pad(inputs[:, 1:], [[0, 0], [0, 1]]),
            "target_length": tf.fill([tf.shape(inputs)[0]],
                                     tf.shape(inputs)[1])
        }

        outputs = []
        next_state = []

        for (model_fn, model_state) in zip(model_fns, state):
            if model_state:
                output, new_state = model_fn(local_features, model_state)
                outputs.append(output)
                next_state.append(new_state)
            else:
                output = model_fn(local_features)
                outputs.append(output)
                next_state.append({})

        # Ensemble
        log_prob = tf.add_n(outputs) / float(len(outputs))

        return log_prob, next_state

    return inference_fn


def _sampling_step(time, func, state, min_length, max_length, pad_id, eos_id):
    # Compute log probabilities
    seqs = state.inputs
    # [batch_size * num_samples, vocab_size]
    step_log_probs, next_state = func(seqs, state.state)

    # Suppress <eos> if needed
    batch_size = tf.shape(step_log_probs)[0]
    vocab_size = step_log_probs.shape[-1].value or tf.shape(step_log_probs)[1]
    add_mask = tf.one_hot(eos_id, vocab_size, dtype=step_log_probs.dtype,
                          on_value=step_log_probs.dtype.min,
                          off_value=0.0)
    add_mask = utils.tile_batch(tf.reshape(add_mask, [1, -1]), batch_size)
    add_mask = tf.where(time < min_length, add_mask, tf.zeros_like(add_mask))
    step_log_probs = step_log_probs + add_mask

    # sample from distribution
    symbol_indices = tf.multinomial(step_log_probs, 1, output_dtype=tf.int32)
    symbol_scores = tf.squeeze(utils.gather_2d(step_log_probs, symbol_indices),
                               axis=1)
    curr_flags = tf.squeeze(tf.equal(symbol_indices, eos_id), axis=1)
    curr_flags = tf.logical_or(state.flags, curr_flags)

    # Append <pad> to finished samples
    symbol_indices = tf.where(state.flags, tf.fill([batch_size, 1], pad_id),
                              symbol_indices)
    symbol_scores = tf.where(state.flags, tf.zeros([batch_size]),
                             symbol_scores)

    # Force sampler to generate <eos> if length exceed max_length
    eos_flags = tf.where(time > max_length, tf.ones([batch_size], tf.bool),
                         tf.zeros([batch_size], tf.bool))
    eos_scores = tf.squeeze(utils.gather_2d(step_log_probs,
                                            tf.fill([batch_size, 1], eos_id)),
                            axis=1)
    eos_indices = tf.fill([batch_size, 1], eos_id)
    cond = tf.logical_and(tf.logical_not(curr_flags), eos_flags)
    curr_flags = tf.logical_or(curr_flags, eos_flags)
    symbol_indices = tf.where(cond, eos_indices, symbol_indices)
    symbol_scores = tf.where(cond, eos_scores, symbol_scores)

    new_state = SamplerState(
        inputs=tf.concat([seqs, symbol_indices], axis=1),
        state=next_state,
        scores=state.scores + symbol_scores,
        flags=curr_flags
    )

    return time + 1, new_state


def random_sample(func, state, batch_size, min_length, max_length, pad_id,
                  bos_id, eos_id):
    init_seqs = tf.fill([batch_size, 1], bos_id)
    init_scores = tf.zeros([batch_size])
    init_flags = tf.zeros([batch_size], tf.bool)

    state = SamplerState(
        inputs=init_seqs,
        state=state,
        scores=init_scores,
        flags=init_flags
    )

    max_step = tf.reduce_max(max_length)

    def _is_finished(t, s):
        all_finished = tf.reduce_all(s.flags)
        cond = tf.logical_and(tf.less(t, max_step),
                              tf.logical_not(all_finished))

        return cond

    def _loop_fn(t, s):
        outs = _sampling_step(t, func, s, min_length, max_length, pad_id,
                              eos_id)
        return outs

    time = tf.constant(0, name="time")
    shape_invariants = SamplerState(
        inputs=tf.TensorShape([None, None]),
        state=nest.map_structure(utils.infer_shape_invariants, state.state),
        scores=tf.TensorShape([None]),
        flags=tf.TensorShape([None])
    )
    outputs = tf.while_loop(_is_finished, _loop_fn, [time, state],
                            shape_invariants=[tf.TensorShape([]),
                                              shape_invariants],
                            parallel_iterations=1,
                            back_prop=False)

    final_state = outputs[1]
    final_seqs = final_state.inputs
    final_scores = final_state.scores

    return final_seqs, final_scores


def create_sampling_graph(models, features, params):
    if not isinstance(models, (list, tuple)):
        raise ValueError("'models' must be a list or tuple")

    features = copy.copy(features)
    model_fns = [model.get_inference_func() for model in models]

    num_samples = params.num_samples

    # Compute initial state if necessary
    states = []
    funcs = []

    for model_fn in model_fns:
        if callable(model_fn):
            # For non-incremental decoding
            states.append({})
            funcs.append(model_fn)
        else:
            # For incremental decoding where model_fn is a tuple:
            # (encoding_fn, decoding_fn)
            states.append(model_fn[0](features))
            funcs.append(model_fn[1])

    batch_size = tf.shape(features["source"])[0]
    pad_id = params.mapping["target"][params.pad]
    bos_id = params.mapping["target"][params.bos]
    eos_id = params.mapping["target"][params.eos]

    # Expand the inputs
    features["source"] = utils.tile_batch(features["source"], num_samples)
    features["source_length"] = utils.tile_batch(features["source_length"],
                                                 num_samples)

    min_length = tf.to_float(features["source_length"])
    max_length = tf.to_float(features["source_length"])

    if params.min_length_ratio:
        min_length = min_length * params.min_length_ratio

    if params.max_length_ratio:
        max_length = max_length * params.max_length_ratio

    if params.min_sample_length:
        min_length = min_length - params.min_sample_length

    if params.max_sample_length:
        max_length = max_length + params.max_sample_length

    min_length = tf.to_int32(min_length)
    max_length = tf.to_int32(max_length)

    decoding_fn = _get_inference_fn(funcs, features)
    states = nest.map_structure(lambda x: utils.tile_batch(x, num_samples),
                                states)

    seqs, scores = random_sample(decoding_fn, states, batch_size * num_samples,
                                 min_length, max_length, pad_id, bos_id,
                                 eos_id)

    seqs = tf.reshape(seqs, [batch_size, num_samples, -1])
    scores = tf.reshape(scores, [batch_size, num_samples])

    return seqs[:, :, 1:], scores
