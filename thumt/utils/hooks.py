# coding=utf-8
# Copyright 2017 The THUMT authors

import tensorflow as tf


class ValidationHook(tf.train.SessionRunHook):

    def __init__(self, val_file, ref_files, score):
        self._val_file = val_file
        self._ref_files = ref_files
        self._score = score

    def _load(self):
        pass

    def begin(self):
        pass

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        pass
