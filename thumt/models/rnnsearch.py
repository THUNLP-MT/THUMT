# coding=utf-8
# Copyright 2017 The THUMT Authors

import copy
import tensorflow as tf
import thumt.layers as layers
import thumt.utils.search as search

from .model import NMTModel


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)


def _gru_encoder(cell, inputs, sequence_length, initial_state, dtype=None):
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state):
        inp_t = input_ta.read(t)
        cell_output, new_state = cell(inp_t, state)
        cell_output = _copy_through(t, sequence_length, zero_output,
                                    cell_output)
        new_state = _copy_through(t, sequence_length, state, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    return all_output, final_state


def _encoder(cell_fw, cell_bw, inputs, sequence_length, dtype=None,
             scope=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length]):
        inputs_fw = inputs
        inputs_bw = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)

        with tf.variable_scope("forward"):
            output_fw, state_fw = _gru_encoder(cell_fw, inputs_fw,
                                               sequence_length, None,
                                               dtype=dtype)

        with tf.variable_scope("backward"):
            output_bw, state_bw = _gru_encoder(cell_bw, inputs_bw,
                                               sequence_length, None,
                                               dtype=dtype)
            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            batch_axis=0, seq_axis=1)

        results = {
            "annotation": tf.concat([output_fw, output_bw], axis=2),
            "outputs": {
                "forward": output_fw,
                "backward": output_bw
            },
            "final_states": {
                "forward": state_fw,
                "backward": state_bw
            }
        }

        return results


def _decoder(cell, inputs, memory, sequence_length, initial_state, dtype=None,
             scope=None):
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])
        mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(memory)[1],
                                    dtype=tf.float32)
        bias = layers.nn.attention_bias(mem_mask)
        #cache = layers.nn.attention(None, memory, None, output_size)

        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
        value_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="value_array")
        alpha_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="alpha_array")
        input_ta = input_ta.unstack(inputs)
        initial_state = layers.nn.linear(initial_state, output_size, True,
                                         scope="s_transform")
        initial_state = tf.tanh(initial_state)

        def loop_func(t, out_ta, att_ta, val_ta, state):
            inp_t = input_ta.read(t)
            results = layers.nn.attention(state, memory, bias, output_size,
                                          cache=None)
            alpha = results["weight"]
            context = results["value"]
            cell_input = [inp_t, context]
            cell_output, new_state = cell(cell_input, state)
            cell_output = _copy_through(t, sequence_length["target"],
                                        zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state,
                                      new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value,
                                      context)

            out_ta = out_ta.write(t, cell_output)
            att_ta = att_ta.write(t, alpha)
            val_ta = val_ta.write(t, new_value)
            return t + 1, out_ta, att_ta, val_ta, new_state

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, alpha_ta, value_ta, initial_state)

        outputs = tf.while_loop(lambda t, *_: t < time_steps,
                                loop_func, loop_vars,
                                parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        value_final_ta = outputs[3]

        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])

        final_value = value_final_ta.stack()
        final_value.set_shape([None, None, memory.shape[-1].value])
        final_value = tf.transpose(final_value, [1, 0, 2])

        result = {
            "outputs": final_output,
            "values": final_value,
            "initial_state": initial_state
        }

    return result


def _encoding_graph(features, params):
    # Encoding and pre-computation
    src_vocab_size = len(params.vocabulary["source"])

    with tf.device("/cpu:0"):
        with tf.variable_scope("source_embedding"):
            src_emb = tf.get_variable("embedding",
                                      [src_vocab_size,
                                       params.embedding_size])
            src_bias = tf.get_variable("bias", [params.embedding_size])
            src_inputs = tf.nn.embedding_lookup(src_emb,
                                                features["source"])
            src_inputs = tf.nn.bias_add(src_inputs, src_bias)

    cell_fw = layers.rnn_cell.LegacyGRUCell(params.hidden_size)
    cell_bw = layers.rnn_cell.LegacyGRUCell(params.hidden_size)

    encoder_output = _encoder(cell_fw, cell_bw, src_inputs,
                              features["source_length"])
    annotation = encoder_output["annotation"]
    initial_state = encoder_output["final_states"]["backward"]

    with tf.variable_scope("decoder"):
        mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(annotation)[1],
                                    dtype=tf.float32)
        bias = layers.nn.attention_bias(mem_mask)
        cache = layers.nn.attention(None, memory, None, output_size)
        initial_state = layers.nn.linear(initial_state, output_size, True,
                                         scope="s_transform")
        initial_state = tf.tanh(initial_state)

    # All features used in decoding must listed here
    # The first dimension of each tensor must be BATCH_DIM
    decoding_features = {
        "annotation": annotation,
        "state": initial_state,
        "weight": tf.zeros_like(annotation[:, :, 0]),
        "attention_bias": bias,
        "attention_cache": cache,
        "attention_vector": tf.zeros_like(annotation[:, 0, :])
    }

    return decoding_features


def _decoding_graph(features, params):
    # Decoding
    # The computation of original RNNsearch's decoding step is confusing
    # * At the initial step, the implementation uses initial state computed
    #   using encoder final state to obtain attention context. Then
    #   combined with initial state and zero embedding to predict next
    #   words
    # * For other steps. The new state is computed by feeding previous
    #   state, current word input and previous context vector. The new
    #   context vector is computed using new state. The probability
    #   distribution is computed using input word embedding, new context
    #   and new state

    tgt_vocab_size = len(params.vocabulary["target"])

    # Special case for initial step
    def initial_step_fn():
        cache = {"key": features["attention_cache"]}

        with tf.variable_scope("decoder"):
            results = layers.nn.attention(features["state"],
                                          features["annotation"],
                                          features["attention_bias"],
                                          params.hidden_size,
                                          cache=cache)
        alpha = results["weight"]
        context = results["value"]
        batch_size = tf.shape(context)[0]
        zero_embedding = tf.zeros([batch_size, params.embedding_size])
        maxout_features = [
            zero_embedding,
            features["state"],
            context
        ]

        readout = layers.nn.maxout(tf.concat(maxout_features, axis=-1),
                                   params.embedding_size)

        # Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True,
                                  scope="softmax")
        prob_dist = tf.nn.softmax(logits)

        # The features must exactly the same with what encoding returned
        decoding_features = {
            "annotation": features["annotation"],
            "state": features["state"],
            "weight": alpha,
            "attention_bias": features["attention_bias"],
            "attention_cache": features["attention_cache"],
            "attention_vector": context
        }

        return prob_dist, decoding_features

    def step_fn():
        with tf.device("/cpu:0"):
            with tf.variable_scope("source_embedding"):
                tgt_emb = tf.get_variable("embedding",
                                          [tgt_vocab_size,
                                           params.embedding_size])
                tgt_bias = tf.get_variable("bias", [params.embedding_size])

                # Take last word
                tgt_inputs = tf.nn.embedding_lookup(
                    tgt_emb,
                    features["target"][:, -1]
                )
                tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)

        # RNN step
        with tf.variable_scope("decoder"):
            cache = {"key": features["attention_cache"]}
            cell = layers.rnn_cell.LegacyGRUCell(params.hidden_size)
            # 1. Compute new state
            context = features["attention_vector"]
            output, new_state = cell([tgt_inputs, context], features["state"])
            # 2. Compute new context
            results = layers.nn.attention(new_state,
                                          features["annotation"],
                                          features["attention_bias"],
                                          params.hidden_size,
                                          cache=cache)
            alpha = results["weight"]
            context = results["value"]
            # 3. Prediction
            maxout_features = [
                tgt_inputs,
                new_state,
                context
            ]

            maxout_size = params.hidden_size / params.maxnum
            maxhid = layers.nn.maxout(maxout_features, maxout_size,
                                      params.maxnum, concat=False)
            readout = layers.nn.linear(maxhid, params.embedding_size, False,
                                       scope="deepout")

            # Prediction
            logits = layers.nn.linear(readout, tgt_vocab_size, True,
                                      scope="softmax")
            prob_dist = tf.nn.softmax(logits)

            # The features must exactly the same with what encoding returned
            decoding_features = {
                "annotation": features["annotation"],
                "state": new_state,
                "weight": alpha,
                "attention_bias": features["attention_bias"],
                "attention_cache": features["attention_cache"],
                "attention_vector": context,
            }

        return prob_dist, decoding_features

    dist, new_features = tf.cond(tf.equal(tf.shape(features["target"])[1], 1),
                                 initial_step_fn, step_fn)

    return dist, new_features


def model_parameters():
    params = tf.contrib.training.HParams(
        # vocabulary
        pad="</s>",
        unk="UNK",
        eos="<eos>",
        bos="</s>",
        append_eos=True,
        # model
        rnn_cell="LegacyGRUCell",
        embedding_size=620,
        hidden_size=1000,
        maxnum=2,
        # regularization
        dropout=0.2,
        use_variational_dropout=False,
        label_smoothing=0.1,
    )

    return params


def training_graph(features, labels, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])

    with tf.device("/cpu:0"):
        with tf.variable_scope("source_embedding"):
            src_emb = tf.get_variable("embedding",
                                      [src_vocab_size, params.embedding_size])
            src_bias = tf.get_variable("bias", [params.embedding_size])
            src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])
            src_inputs = tf.nn.bias_add(src_inputs, src_bias)

        with tf.variable_scope("target_embedding"):
            tgt_emb = tf.get_variable("embedding",
                                      [tgt_vocab_size, params.embedding_size])
            tgt_bias = tf.get_variable("bias", [params.embedding_size])
            tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])
            tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)

    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
        tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

    # encoder
    cell_fw = layers.rnn_cell.LegacyGRUCell(params.hidden_size)
    cell_bw = layers.rnn_cell.LegacyGRUCell(params.hidden_size)

    if params.use_variational_dropout:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )

    encoder_output = _encoder(cell_fw, cell_bw, src_inputs,
                              features["source_length"])

    # decoder
    cell = layers.rnn_cell.LegacyGRUCell(params.hidden_size)

    if params.use_variational_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            # input + context
            input_size=params.embedding_size + 2 * params.hidden_size,
            dtype=tf.float32
        )

    length = {
        "source": features["source_length"],
        "target": features["target_length"]
    }
    initial_state = encoder_output["final_states"]["backward"]
    decoder_output = _decoder(cell, tgt_inputs, encoder_output["annotation"],
                              length, initial_state)

    # Shift left
    shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
    shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]

    all_outputs = tf.concat(
        [
            tf.expand_dims(decoder_output["initial_state"], axis=1),
            decoder_output["outputs"],
        ],
        axis=1
    )
    shifted_outputs = all_outputs[:, :-1, :]

    maxout_features = [
        shifted_tgt_inputs,
        shifted_outputs,
        decoder_output["values"]
    ]
    maxout_size = params.hidden_size / params.maxnum

    if labels is None:
        # Special case for non-incremental decoding
        maxout_features = [
            shifted_tgt_inputs[:, -1, :],
            shifted_outputs[:, -1, :],
            decoder_output["values"][:, -1, :]
        ]
        maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                                  concat=False)
        readout = layers.nn.linear(maxhid, params.embedding_size, False,
                                   scope="deepout")

        # Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True,
                                  scope="softmax")

        return logits

    maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                              concat=False)
    readout = layers.nn.linear(maxhid, params.embedding_size, False,
                               scope="deepout")

    if params.dropout and not params.use_variational_dropout:
        readout = tf.nn.dropout(readout, 1.0 - params.dropout)

    # Prediction
    logits = layers.nn.linear(readout, tgt_vocab_size, True, scope="softmax")
    logits = tf.reshape(logits, [-1, tgt_vocab_size])

    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        label_smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(labels))
    tgt_mask = tf.to_float(tf.sequence_mask(features["target_length"]))
    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    results = {
        "loss": loss,
        "logits": logits
    }

    return results


def incremental_inference_graph(features, params):
    if "target" not in features:
        return _encoding_graph(features, params)
    else:
        return _decoding_graph(features, params)


class RNNsearch(NMTModel):

    def __init__(self, params, scope="rnnsearch"):
        super(RNNsearch, self).__init__(params=params, scope=scope)

    def build_training_graph(self, features, initializer):
        with tf.variable_scope(self._scope, initializer=initializer):
            results = training_graph(features, features["target"],
                                     self.parameters)
            return results["loss"]

    def build_evaluation_graph(self, features):
        raise NotImplementedError("Not implemented")

    def build_inference_graph(self, features):
        # This function is costly.
        params = copy.copy(self.parameters)
        params.dropout = 0.0
        params.use_variational_dropout = False,
        params.label_smoothing = 0.0

        with tf.variable_scope(self._scope):
            results = search.create_inference_graph(
                lambda f, p: training_graph(f, None, p), features, params
            )

        return results

    def build_incremental_decoder(self):
        # This is the preferred method to be called during inference stage
        raise NotImplementedError("Not implemented")

    @staticmethod
    def model_parameters():
        return model_parameters()
