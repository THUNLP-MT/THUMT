# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from thumt.layers.nn import linear


class LegacyGRUCell(tf.nn.rnn_cell.RNNCell):
    """ Groundhog's implementation of GRUCell

    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None):
        super(LegacyGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            all_inputs = list(inputs) + [state]
            r = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="reset_gate"))
            u = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="update_gate"))
            all_inputs = list(inputs) + [r * state]
            c = linear(all_inputs, self._num_units, True, False,
                       scope="candidate")

            new_state = (1.0 - u) * state + u * tf.tanh(c)

        return new_state, new_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
