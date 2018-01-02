# coding=utf-8
# Code modified from Tensor2Tensor library
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# Default value for INF
INF = 1. * 1e7


def log_prob_from_logits(logits):
    return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)


def compute_batch_indices(batch_size, beam_size):
    """Computes the i'th coordinate that contains the batch index for gathers.

    Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    batch the beam item is in. This will create the i of the i,j coordinate
    needed for the gather.

    :param batch_size: Batch size
    :param beam_size: Size of the beam.
    :returns: batch_pos: [batch_size, beam_size] tensor of ids
    """
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos


def compute_topk_scores_and_seq(sequences, scores, scores_to_gather, flags,
                                beam_size, batch_size):
    """Given sequences and scores, will gather the top k=beam size sequences.

    This function is used to grow alive, and finished. It takes sequences,
    scores, and flags, and returns the top k from sequences, scores_to_gather,
    and flags based on the values in scores.

    :param sequences: Tensor of sequences that we need to gather from.
        [batch_size, beam_size, seq_length]
    :param scores: Tensor of scores for each sequence in sequences.
        [batch_size, beam_size]. We will use these to compute the topk.
    :param scores_to_gather: Tensor of scores for each sequence in sequences.
        [batch_size, beam_size]. We will return the gathered scores from
        here. Scores to gather is different from scores because for
        grow_alive, we will need to return log_probs, while for
        grow_finished, we will need to return the length penalized scors.
    :param flags: Tensor of bools for sequences that say whether a sequence has
        reached EOS or not
    :param beam_size: int
    :param batch_size: int
    :returns: Tuple of (topk_seq [batch_size, beam_size, decode_length],
        topk_gathered_scores [batch_size, beam_size],
        topk_finished_flags[batch_size, beam_size])
    """
    _, topk_indexes = tf.nn.top_k(scores, k=beam_size)
    # The next three steps are to create coordinates for tf.gather_nd to pull
    # out the top-k sequences from sequences based on scores.
    # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    # batch the beam item is in. This will create the i of the i,j coordinate
    # needed for the gather
    batch_pos = compute_batch_indices(batch_size, beam_size)

    # top coordinates will give us the actual coordinates to do the gather.
    # stacking will create a tensor of dimension batch * beam * 2, where the
    # last dimension contains the i,j gathering coordinates.
    top_coordinates = tf.stack([batch_pos, topk_indexes], axis=2)

    # Gather up the highest scoring sequences
    topk_seq = tf.gather_nd(sequences, top_coordinates)
    topk_flags = tf.gather_nd(flags, top_coordinates)
    topk_gathered_scores = tf.gather_nd(scores_to_gather, top_coordinates)
    return topk_seq, topk_gathered_scores, topk_flags


def beam_search(symbols_to_logits_fn, initial_ids, beam_size, decode_length,
                vocab_size, alpha, eos_id, lp_constant=5.0):
    """Beam search with length penalties.

    Uses an interface specific to the sequence cnn models;
    Requires a function that can take the currently decoded symbols and return
    the logits for the next symbol. The implementation is inspired by
    https://arxiv.org/abs/1609.08144.

    :param symbols_to_logits_fn: Interface to the model, to provide logits.
        Should take [batch_size, decoded_ids] and return [
        batch_size, vocab_size]
    :param initial_ids: Ids to start off the decoding, this will be the first
        thing handed to symbols_to_logits_fn
        (after expanding to beam size) [batch_size]
    :param beam_size: Size of the beam.
    :param decode_length: Number of steps to decode for.
    :param vocab_size: Size of the vocab, must equal the size of the logits
        returned by symbols_to_logits_fn
    :param alpha: alpha for length penalty.
    :param eos_id: ID for end of sentence.
    :param lp_constant: A floating number used in length penalty
    :returns: Tuple of (decoded beams [batch_size, beam_size, decode_length]
        decoding probabilities [batch_size, beam_size])
    """
    batch_size = tf.shape(initial_ids)[0]

    # Assume initial_ids are prob 1.0
    initial_log_probs = tf.constant([[0.] + [-float("inf")] * (beam_size - 1)])
    # Expand to beam_size (batch_size, beam_size)
    alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])

    # Expand each batch to beam_size
    alive_seq = tf.tile(tf.expand_dims(initial_ids, 1), [1, beam_size])
    alive_seq = tf.expand_dims(alive_seq, 2)  # (batch_size, beam_size, 1)

    # Finished will keep track of all the sequences that have finished so far
    # Finished log probs will be negative infinity in the beginning
    # finished_flags will keep track of booleans
    finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)
    # Setting the scores of the initial to negative infinity.
    finished_scores = tf.ones([batch_size, beam_size]) * -INF
    finished_flags = tf.zeros([batch_size, beam_size], tf.bool)

    def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                      curr_scores, curr_finished):
        """Given sequences and scores, will gather the top k=beam size
           sequences.

        :param finished_seq: Current finished sequences.
            [batch_size, beam_size, current_decoded_length]
        :param finished_scores: scores for each of these sequences.
            [batch_size, beam_size]
        :param finished_flags: finished bools for each of these sequences.
            [batch_size, beam_size]
        :param curr_seq: current topk sequence that has been grown by one
            position. [batch_size, beam_size, current_decoded_length]
        :param curr_scores: scores for each of these sequences.
            [batch_size, beam_size]
        :param curr_finished: Finished flags for each of these sequences.
            [batch_size, beam_size]
        :returns:
            Tuple of
                (Top-k sequences based on scores,
                log probs of these sequences,
                Finished flags of these sequences)
        """
        # First append a column of 0'ids to finished to make the same length
        # with finished scores
        finished_seq = tf.concat(
            [finished_seq, tf.zeros([batch_size, beam_size, 1], tf.int32)],
            axis=2
        )

        # Set the scores of the unfinished seq in curr_seq to large negative
        # values
        curr_scores += (1. - tf.to_float(curr_finished)) * -INF
        # concatenating the sequences and scores along beam axis
        curr_finished_seq = tf.concat([finished_seq, curr_seq], axis=1)
        curr_finished_scores = tf.concat([finished_scores, curr_scores],
                                         axis=1)
        curr_finished_flags = tf.concat([finished_flags, curr_finished],
                                        axis=1)
        return compute_topk_scores_and_seq(
                curr_finished_seq, curr_finished_scores, curr_finished_scores,
                curr_finished_flags, beam_size, batch_size)

    def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished):
        """Given sequences and scores, will gather the top k=beam size
           sequences.

        :param curr_seq: current topk sequence that has been grown by one
            position. [batch_size, beam_size, i+1]
        :param curr_scores: scores for each of these sequences.
            [batch_size, beam_size]
        :param curr_log_probs: log probs for each of these sequences.
            [batch_size, beam_size]
        :param curr_finished: Finished flags for each of these sequences.
            [batch_size, beam_size]
        :returns:
            Tuple of
                (Top-k sequences based on scores,
                log probs of these sequences,
                Finished flags of these sequences)
        """
        # Set the scores of the finished seq in curr_seq to large negative
        # values
        curr_scores += tf.to_float(curr_finished) * -INF
        return compute_topk_scores_and_seq(curr_seq, curr_scores,
                                           curr_log_probs, curr_finished,
                                           beam_size, batch_size)

    def grow_topk(i, alive_seq, alive_log_probs):
        r"""Inner beam search loop.

        This function takes the current alive sequences, and grows them to
        topk sequences where k = 2*beam. We use 2*beam because, we could have
        beam_size number of sequences that might hit <eos> and there will be
        no alive sequences to continue. With 2*beam_size, this will not happen.
        This relies on the assumption the vocab size is > beam size.
        If this is true, we'll have at least beam_size non <eos> extensions if
        we extract the next top 2*beam words.
        Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
        https://arxiv.org/abs/1609.08144.

        :param i: loop index
        :param alive_seq: Topk sequences decoded so far
            [batch_size, beam_size, i+1]
        :param alive_log_probs: probabilities of these sequences.
            [batch_size, beam_size]
        :returns:
            Tuple of
                (Top-k sequences extended by the next word,
                The log probs of these sequences,
                The scores with length penalty of these sequences,
                Flags indicating which of these sequences have finished
                decoding)
        """
        # Get the logits for all the possible next symbols
        flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])

        # (batch_size * beam_size, decoded_length)
        flat_logits_list = symbols_to_logits_fn(flat_ids)
        logits_list = [
            tf.reshape(flat_logits, (batch_size, beam_size, -1))
            for flat_logits in flat_logits_list
        ]

        # Convert logits to normalized log probs
        candidate_log_probs = [
            log_prob_from_logits(logits)
            for logits in logits_list
        ]

        n_models = len(candidate_log_probs)
        candidate_log_probs = tf.add_n(candidate_log_probs) / float(n_models)

        # Multiply the probabilities by the current probabilities of the beam.
        # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
        log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs,
                                                         axis=2)

        length_penalty = tf.pow(
            ((lp_constant + tf.to_float(i + 1)) / (1.0 + lp_constant)), alpha
        )

        curr_scores = log_probs / length_penalty
        # Flatten out (beam_size, vocab_size) probs in to a list of
        # possibilities
        flat_curr_scores = tf.reshape(curr_scores,
                                      [-1, beam_size * vocab_size])

        topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size * 2)

        # Recovering the log probs because we will need to send them back
        topk_log_probs = topk_scores * length_penalty

        # Work out what beam the top probs are in.
        topk_beam_index = topk_ids // vocab_size
        topk_ids %= vocab_size  # Unflatten the ids

        # The next three steps are to create coordinates for tf.gather_nd to
        # pull
        # out the correct sequences from id's that we need to grow.
        # We will also use the coordinates to gather the booleans of the beam
        # items that survived.
        batch_pos = compute_batch_indices(batch_size, beam_size * 2)

        # top beams will give us the actual coordinates to do the gather.
        # stacking will create a tensor of dimension batch * beam * 2, where
        # the last dimension contains the i,j gathering coordinates.
        topk_coordinates = tf.stack([batch_pos, topk_beam_index], axis=2)

        # Gather up the most probable 2*beams both for the ids and
        # finished_in_alive bools
        topk_seq = tf.gather_nd(alive_seq, topk_coordinates)

        # Append the most probable alive
        topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)],
                              axis=2)

        topk_finished = tf.equal(topk_ids, eos_id)

        return topk_seq, topk_log_probs, topk_scores, topk_finished

    def inner_loop(i, alive_seq, alive_log_probs, finished_seq,
                   finished_scores, finished_flags):
        """Inner beam search loop.

        There are three groups of tensors, alive, finished, and topk.
        The alive group contains information about the current alive sequences
        The top-k group contains information about alive + topk current decoded
        words the finished group contains information about finished sentences,
        that is, the ones that have decoded to <EOS>. These are what we return.
        The general beam search algorithm is as follows:
        While we haven't terminated (pls look at termination condition)
            1. Grow the current alive to get beam*2 top-k sequences
            2. Among the top-k, keep the top beam_size ones that haven't
            reached <eos> into alive
            3. Among the top-k, keep the top beam_size ones have reached <eos>
            into finished
        Repeat
        To make things simple with using fixed size tensors, we will end
        up inserting unfinished sequences into finished in the beginning. To
        stop that we add -ve INF to the score of the unfinished sequence so
        that when a true finished sequence does appear, it will have a higher
        score than all the unfinished ones.

        :param i: loop index
        :param alive_seq: Topk sequences decoded so far
            [batch_size, beam_size, i+1]
        :param alive_log_probs: probabilities of the beams.
            [batch_size, beam_size]
        :param finished_seq: Current finished sequences.
            [batch_size, beam_size, i+1]
        :param finished_scores: scores for each of these sequences.
            [batch_size, beam_size]
        :param finished_flags: finished bools for each of these sequences.
            [batch_size, beam_size]

        :returns:
            Tuple of
                (Incremented loop index
                New alive sequences,
                Log probs of the alive sequences,
                New finished sequences,
                Scores of the new finished sequences,
                Flags indicating which sequence in finished as reached EOS)
        """

        # Each inner loop, we carry out three steps:
        # 1. Get the current topk items.
        # 2. Extract the ones that have finished and haven't finished
        # 3. Recompute the contents of finished based on scores.
        topk_seq, topk_log_probs, topk_scores, topk_finished = grow_topk(
            i, alive_seq, alive_log_probs
        )
        alive_seq, alive_log_probs, _ = grow_alive(topk_seq, topk_scores,
                                                   topk_log_probs,
                                                   topk_finished)
        finished_seq, finished_scores, finished_flags = grow_finished(
            finished_seq, finished_scores, finished_flags, topk_seq,
            topk_scores, topk_finished
        )

        return (i + 1, alive_seq, alive_log_probs, finished_seq,
                finished_scores, finished_flags)

    def _is_finished(i, unused_alive_seq, alive_log_probs, unused_finished_seq,
                     finished_scores, finished_in_finished):
        """Checking termination condition.

        We terminate when we decoded up to decode_length or the lowest scoring
        item in finished has a greater score that the highest prob item in
        alive divided by the max length penalty

        :param i: loop index
        :param alive_log_probs: probabilities of the beams.
            [batch_size, beam_size]
        :param finished_scores: scores for each of these sequences.
            [batch_size, beam_size]
        :param finished_in_finished: finished bools for each of these
            sequences. [batch_size, beam_size]

        :returns: Bool.
        """
        max_length_penalty = tf.pow(((5. + tf.to_float(decode_length)) / 6.),
                                    alpha)
        # The best possible score of the most likley alive sequence
        lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

        # Now to compute the lowest score of a finished sequence in finished
        # If the sequence isn't finished, we multiply it's score by 0. since
        # scores are all -ve, taking the min will give us the score of the
        # lowest finished item.
        lowest_score_of_finished_in_finished = tf.reduce_min(
            finished_scores * tf.to_float(finished_in_finished), axis=1
        )
        # If none of the sequences have finished, then the min will be 0 and
        # we have to replace it by -ve INF if it is. The score of any seq in
        # alive will be much higher than -ve INF and the termination condition
        # will not be met.
        lowest_score_of_finished_in_finished += (
            (1. - tf.to_float(tf.reduce_any(finished_in_finished, 1))) * -INF
        )

        bound_is_met = tf.reduce_all(
            tf.greater(lowest_score_of_finished_in_finished,
                       lower_bound_alive_scores)
        )

        return tf.logical_and(tf.less(i, decode_length),
                              tf.logical_not(bound_is_met))

    (_, alive_seq, alive_log_probs, finished_seq, finished_scores,
     finished_flags) = tf.while_loop(
        _is_finished,
        inner_loop, [
            tf.constant(0), alive_seq, alive_log_probs, finished_seq,
            finished_scores, finished_flags
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, None, None]),
            alive_log_probs.get_shape(),
            tf.TensorShape([None, None, None]),
            finished_scores.get_shape(),
            finished_flags.get_shape()
        ],
        parallel_iterations=1,
        back_prop=False
    )

    alive_seq.set_shape((None, beam_size, None))
    finished_seq.set_shape((None, beam_size, None))

    # Accounting for corner case: It's possible that no sequence in alive for
    # a particular batch item ever reached <eos>. In that case, we should just
    # copy the contents of alive for that batch item.
    # tf.reduce_any(finished_flags, 1) if 0, means that no sequence for that
    # batch index had reached EOS. We need to do the same for the scores as
    # well.
    finished_seq = tf.where(
            tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
    finished_scores = tf.where(
            tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
    return finished_seq, finished_scores


def create_inference_graph(model_fns, features, params):
    if not isinstance(model_fns, (list, tuple)):
        model_fns = [model_fns]

    decode_length = params.decode_length
    beam_size = params.beam_size
    top_beams = params.top_beams
    alpha = params.decode_alpha

    # [batch, decoded_ids] => [batch, vocab_size]
    def symbols_to_logits_fn(decoded_ids):
        features["target"] = tf.pad(decoded_ids[:, 1:], [[0, 0], [0, 1]])
        features["target_length"] = tf.fill([tf.shape(features["target"])[0]],
                                            tf.shape(features["target"])[1])

        results = []

        for i, model_fn in enumerate(model_fns):
            results.append(model_fn(features))

        return results

    batch_size = tf.shape(features["source"])[0]
    # Prepend <bos> symbol
    bos_id = params.mapping["target"][params.bos]
    initial_ids = tf.fill([batch_size], tf.constant(bos_id, dtype=tf.int32))

    inputs_old = features["source"]
    inputs_length_old = features["source_length"]

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

    # [batch, beam_size, length] => [batch * beam_size, length]
    features["source_length"] = tf.reshape(features["source_length"],
                                           [shape[0] * shape[1]])

    vocab_size = len(params.vocabulary["target"])
    # Setting decode length to input length + decode_length
    decode_length = tf.shape(features["source"])[1] + decode_length

    ids, scores = beam_search(symbols_to_logits_fn, initial_ids,
                              beam_size, decode_length, vocab_size,
                              alpha,
                              eos_id=params.mapping["target"][params.eos],
                              lp_constant=params.decode_constant)

    # Set inputs back to the unexpanded inputs to not to confuse the Estimator
    features["source"] = inputs_old
    features["source_length"] = inputs_length_old

    # Return `top_beams` decoding
    # (also remove initial id from the beam search)
    if not params.decode_normalize:
        if top_beams == 1:
            return ids[:, 0, 1:]
        else:
            return ids[:, :top_beams, 1:]
    else:
        if top_beams == 1:
            return ids[:, 0, 1:], scores[:, 0]
        else:
            return ids[:, :top_beams, 1:], scores[:, :top_beams]
