# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from collections import namedtuple
from tensorflow.python.util import nest


class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "state", "finish"))):
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


def _infer_shape(x):
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.shape.dims is None:
        return tf.shape(x)

    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    ret = []
    for i in range(len(static_shape)):
        dim = static_shape[i]
        if dim is None:
            dim = dynamic_shape[i]
        ret.append(dim)

    return ret


def _infer_shape_invariants(tensor):
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


def _merge_first_two_dims(tensor):
    shape = _infer_shape(tensor)
    shape[0] *= shape[1]
    shape.pop(1)
    return tf.reshape(tensor, shape)


def _split_first_two_dims(tensor, dim_0, dim_1):
    shape = _infer_shape(tensor)
    new_shape = [dim_0] + [dim_1] + shape[1:]
    return tf.reshape(tensor, new_shape)


def _tile_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size. """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def _gather_2d(params, indices, name=None):
    """ Gather the 2nd dimension given indices
    :param params: A tensor with shape [batch_size, M, ...]
    :param indices: A tensor with shape [batch_size, N]
    :return: A tensor with shape [batch_size, N, ...]
    """
    batch_size = tf.shape(params)[0]
    range_size = tf.shape(indices)[1]
    batch_pos = tf.range(batch_size * range_size) // range_size
    batch_pos = tf.reshape(batch_pos, [batch_size, range_size])
    indices = tf.stack([batch_pos, indices], axis=-1)
    output = tf.gather_nd(params, indices, name=name)

    return output


def _beam_search_step(time, func, state, batch_size, beam_size, alpha,
                      pad_id, eos_id):
    # Compute log probabilities
    seqs, log_probs = state.inputs[:2]
    flat_seqs = _merge_first_two_dims(seqs)
    flat_state = nest.map_structure(lambda x: _merge_first_two_dims(x),
                                    state.state)
    step_log_probs, next_state = func(flat_seqs, flat_state)
    step_log_probs = _split_first_two_dims(step_log_probs, batch_size,
                                           beam_size)
    next_state = nest.map_structure(
        lambda x: _split_first_two_dims(x, batch_size, beam_size), next_state)
    curr_log_probs = tf.expand_dims(log_probs, 2) + step_log_probs

    # Apply length penalty
    length_penalty = tf.pow((5.0 + tf.to_float(time + 1)) / 6.0, alpha)
    curr_scores = curr_log_probs / length_penalty
    vocab_size = curr_scores.shape[-1].value or tf.shape(curr_scores)[-1]

    # Select top-k candidates
    # [batch_size, beam_size * vocab_size]
    curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size])
    # [batch_size, 2 * beam_size]
    top_scores, top_indices = tf.nn.top_k(curr_scores, k=2 * beam_size)
    # Shape: [batch_size, 2 * beam_size]
    beam_indices = top_indices // vocab_size
    symbol_indices = top_indices % vocab_size
    # Expand sequences
    # [batch_size, 2 * beam_size, time]
    candidate_seqs = _gather_2d(seqs, beam_indices)
    candidate_seqs = tf.concat([candidate_seqs,
                                tf.expand_dims(symbol_indices, 2)], 2)

    # Expand sequences
    # Suppress finished sequences
    flags = tf.equal(symbol_indices, eos_id)
    # [batch, 2 * beam_size]
    alive_scores = top_scores + tf.to_float(flags) * tf.float32.min
    # [batch, beam_size]
    alive_scores, alive_indices = tf.nn.top_k(alive_scores, beam_size)
    alive_symbols = _gather_2d(symbol_indices, alive_indices)
    alive_indices = _gather_2d(beam_indices, alive_indices)
    alive_seqs = _gather_2d(seqs, alive_indices)
    # [batch_size, beam_size, time + 1]
    alive_seqs = tf.concat([alive_seqs, tf.expand_dims(alive_symbols, 2)], 2)
    alive_state = nest.map_structure(lambda x: _gather_2d(x, alive_indices),
                                     next_state)
    alive_log_probs = alive_scores * length_penalty

    # Select finished sequences
    prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish
    # [batch, 2 * beam_size]
    step_fin_scores = top_scores + (1.0 - tf.to_float(flags)) * tf.float32.min
    # [batch, 3 * beam_size]
    fin_flags = tf.concat([prev_fin_flags, flags], axis=1)
    fin_scores = tf.concat([prev_fin_scores, step_fin_scores], axis=1)
    # [batch, beam_size]
    fin_scores, fin_indices = tf.nn.top_k(fin_scores, beam_size)
    fin_flags = _gather_2d(fin_flags, fin_indices)
    pad_seqs = tf.fill([batch_size, beam_size, 1],
                       tf.constant(pad_id, tf.int32))
    prev_fin_seqs = tf.concat([prev_fin_seqs, pad_seqs], axis=2)
    fin_seqs = tf.concat([prev_fin_seqs, candidate_seqs], axis=1)
    fin_seqs = _gather_2d(fin_seqs, fin_indices)

    new_state = BeamSearchState(
        inputs=(alive_seqs, alive_log_probs, alive_scores),
        state=alive_state,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    return time + 1, new_state


def beam_search(func, state, batch_size, beam_size, max_length, alpha,
                pad_id, bos_id, eos_id):
    init_seqs = tf.fill([batch_size, beam_size, 1], bos_id)
    init_log_probs = tf.constant([[0.] + [tf.float32.min] * (beam_size - 1)])
    init_log_probs = tf.tile(init_log_probs, [batch_size, 1])
    init_scores = tf.zeros_like(init_log_probs)
    fin_seqs = tf.zeros([batch_size, beam_size, 1], tf.int32)
    fin_scores = tf.fill([batch_size, beam_size], tf.float32.min)
    fin_flags = tf.zeros([batch_size, beam_size], tf.bool)

    state = BeamSearchState(
        inputs=(init_seqs, init_log_probs, init_scores),
        state=state,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    max_step = tf.reduce_max(max_length)

    def _is_finished(t, s):
        log_probs = s.inputs[1]
        finished_flags = s.finish[0]
        finished_scores = s.finish[2]
        max_lp = tf.pow(((5.0 + tf.to_float(max_step)) / 6.0), alpha)
        best_alive_score = log_probs[:, 0] / max_lp
        worst_finished_score = tf.reduce_min(
            finished_scores * tf.to_float(finished_flags), axis=1)
        add_mask = 1.0 - tf.to_float(tf.reduce_any(finished_flags, 1))
        worst_finished_score += tf.float32.min * add_mask
        bound_is_met = tf.reduce_all(tf.greater(worst_finished_score,
                                                best_alive_score))

        cond = tf.logical_and(tf.less(t, max_step),
                              tf.logical_not(bound_is_met))

        return cond

    def _loop_fn(t, s):
        outs = _beam_search_step(t, func, s, batch_size, beam_size, alpha,
                                 pad_id, eos_id)
        return outs

    time = tf.constant(0, name="time")
    shape_invariants = BeamSearchState(
        inputs=(tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None])),
        state=nest.map_structure(_infer_shape_invariants, state.state),
        finish=(tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]))
    )
    outputs = tf.while_loop(_is_finished, _loop_fn, [time, state],
                            shape_invariants=[tf.TensorShape([]),
                                              shape_invariants],
                            parallel_iterations=1,
                            back_prop=False)

    final_state = outputs[1]
    alive_seqs = final_state.inputs[0]
    alive_scores = final_state.inputs[2]
    final_flags = final_state.finish[0]
    final_seqs = final_state.finish[1]
    final_scores = final_state.finish[2]

    alive_seqs.set_shape([None, beam_size, None])
    final_seqs.set_shape((None, beam_size, None))

    final_seqs = tf.where(tf.reduce_any(final_flags, 1), final_seqs,
                          alive_seqs)
    final_scores = tf.where(tf.reduce_any(final_flags, 1), final_scores,
                            alive_scores)

    return final_seqs, final_scores


def create_inference_graph(model_fns, features, params):
    if not isinstance(model_fns, (list, tuple)):
        raise ValueError("mode_fns must be a list or tuple")

    features = copy.copy(features)

    decode_length = params.decode_length
    beam_size = params.beam_size
    top_beams = params.top_beams
    alpha = params.decode_alpha

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

    # Expand the inputs in to the beam size
    # [batch, length] => [batch, beam_size, length]
    features["source"] = tf.expand_dims(features["source"], 1)
    features["source"] = tf.tile(features["source"], [1, beam_size, 1])
    shape = tf.shape(features["source"])

    # [batch, beam_size, length] => [batch * beam_size, length]
    features["source"] = tf.reshape(features["source"],
                                    [shape[0] * shape[1], shape[2]])

    # For source sequence length
    features["source_length"] = tf.expand_dims(features["source_length"], 1)
    features["source_length"] = tf.tile(features["source_length"],
                                        [1, beam_size])
    shape = tf.shape(features["source_length"])

    max_length = features["source_length"] + decode_length

    # [batch, beam_size, length] => [batch * beam_size, length]
    features["source_length"] = tf.reshape(features["source_length"],
                                           [shape[0] * shape[1]])
    decoding_fn = _get_inference_fn(funcs, features)
    states = nest.map_structure(lambda x: _tile_to_beam_size(x, beam_size),
                                states)

    seqs, scores = beam_search(decoding_fn, states, batch_size, beam_size,
                               max_length, alpha, pad_id, bos_id, eos_id)

    return seqs[:, :top_beams, 1:], scores[:, :top_beams]
