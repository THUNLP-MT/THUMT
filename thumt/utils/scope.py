# coding=utf-8
# Copyright 2017-2019 The THUMT Authors
# Modified from TensorFlow (tf.name_scope)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import contextlib

# global variable
_NAME_STACK = ""
_NAMES_IN_USE = {}
_VALID_OP_NAME_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$")
_VALID_SCOPE_NAME_REGEX = re.compile("^[A-Za-z0-9_.\\-/]*$")


def unique_name(name, mark_as_used=True):
    global _NAME_STACK

    if _NAME_STACK:
        name = _NAME_STACK + "/" + name

    i = _NAMES_IN_USE.get(name, 0)

    if mark_as_used:
        _NAMES_IN_USE[name] = i + 1

    if i > 0:
        base_name = name

        while name in _NAMES_IN_USE:
            name = "%s_%d" % (base_name, i)
            i += 1

        if mark_as_used:
            _NAMES_IN_USE[name] = 1

    return name


@contextlib.contextmanager
def scope(name):
    global _NAME_STACK

    if name:
        if _NAME_STACK:
            # check name
            if not _VALID_SCOPE_NAME_REGEX.match(name):
                raise ValueError("'%s' is not a valid scope name" % name)
        else:
            # check name strictly
            if not _VALID_OP_NAME_REGEX.match(name):
                raise ValueError("'%s' is not a valid scope name" % name)

    try:
        old_stack = _NAME_STACK

        if not name:
            new_stack = None
        elif name and name[-1] == "/":
            new_stack = name[:-1]
        else:
            new_stack = unique_name(name)

        _NAME_STACK = new_stack

        yield "" if new_stack is None else new_stack + "/"
    finally:
        _NAME_STACK = old_stack


def get_scope():
    return _NAME_STACK
