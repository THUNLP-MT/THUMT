# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def session_run(monitored_session, args):
    # Call raw TF session directly
    return monitored_session._tf_sess().run(args)
