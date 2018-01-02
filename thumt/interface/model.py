# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class NMTModel(object):

    def __init__(self, params, scope):
        self._scope = scope
        self._params = params

    def get_training_func(self, initializer):
        raise NotImplementedError("Not implemented")

    def get_evaluation_func(self):
        raise NotImplementedError("Not implemented")

    def get_inference_func(self):
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
