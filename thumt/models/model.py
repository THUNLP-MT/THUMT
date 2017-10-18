# coding=utf-8
# Copyright 2017 The THUMT Authors


class NMTModel(object):

    def __init__(self, params, scope):
        self._scope = scope
        self._params = params

    def build_training_graph(self, features, initializer):
        raise NotImplementedError("Not implemented")

    def build_evaluation_graph(self, features):
        raise NotImplementedError("Not implemented")

    def build_inference_graph(self, features):
        raise NotImplementedError("Not implemented")

    def build_incremental_decoder(self):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def model_parameters():
        raise NotImplementedError("Not implemented")

    @property
    def parameters(self):
        return self._params


class NMTIncrementalDecoder(object):
    pass
