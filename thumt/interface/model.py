# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class NMTModel(object):
    """ Abstract object representing an NMT model """

    def __init__(self, params, scope):
        self._scope = scope
        self._params = params

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        """
        :param initializer: the initializer used to initialize the model
        :param regularizer: the regularizer used for model regularization
        :param dtype: an instance of tf.DType
        :return: a function with the following signature:
            (features, params, reuse) -> loss
        """
        raise NotImplementedError("Not implemented")

    def get_evaluation_func(self):
        """
        :return: a function with the following signature:
            (features, params) -> score
        """
        raise NotImplementedError("Not implemented")

    def get_inference_func(self):
        """
        :returns:
            If a model implements incremental decoding, this function should
            returns a tuple of (encoding_fn, decoding_fn), with the following
            requirements:
                encoding_fn: (features, params) -> initial_state
                decoding_fn: (feature, state, params) -> log_prob, next_state

            If a model does not implement the incremental decoding (slower
            decoding speed but easier to write the code), then this
            function should returns a single function with the following
            signature:
                (features, params) -> log_prob

            See models/transformer.py and models/rnnsearch.py
            for comparison.
        """
        raise NotImplementedError("Not implemented")

    @staticmethod
    def get_name():
        raise NotImplementedError("Not implemented")

    @staticmethod
    def get_parameters():
        raise NotImplementedError("Not implemented")

    @property
    def parameters(self):
        return self._params
