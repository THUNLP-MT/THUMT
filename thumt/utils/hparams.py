# coding=utf-8
# Copyright 2017-2019 The THUMT Authors
# Modified from TensorFlow (tf.contrib.training.HParams)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import re
import six


def parse_values(values, type_map):
    ret = {}
    param_re = re.compile(r"(?P<name>[a-zA-Z][\w]*)\s*=\s*"
                          r"((?P<val>[^,\[]*)|\[(?P<vals>[^\]]*)\])($|,)")
    pos = 0

    while pos < len(values):
        m = param_re.match(values, pos)

        if not m:
            raise ValueError(
                "Malformed hyperparameter value: %s" % values[pos:])

        # Check that there is a comma between parameters and move past it.
        pos = m.end()
        # Parse the values.
        m_dict = m.groupdict()
        name = m_dict["name"]

        if name not in type_map:
            raise ValueError("Unknown hyperparameter type for %s" % name)

        def parse_fail():
            raise ValueError("Could not parse hparam %s in %s" % (name, values))

        if type_map[name] == bool:
            def parse_bool(value):
                if value == "true":
                    return True
                elif value == "false":
                    return False
                else:
                    try:
                        return bool(int(value))
                    except ValueError:
                        parse_fail()
            parse = parse_bool
        else:
            parse = type_map[name]


        if m_dict["val"] is not None:
            try:
                ret[name] = parse(m_dict["val"])
            except ValueError:
                parse_fail()
        elif m_dict["vals"] is not None:
            elements = filter(None, re.split("[ ,]", m_dict["vals"]))
            try:
                ret[name] = [parse(e) for e in elements]
            except ValueError:
                parse_fail()
        else:
            parse_fail()

    return ret


class HParams(object):

    def __init__(self, **kwargs):
        self._hparam_types = {}

        for name, value in six.iteritems(kwargs):
            self.add_hparam(name, value)

    def add_hparam(self, name, value):
        if getattr(self, name, None) is not None:
            raise ValueError("Hyperparameter name is reserved: %s" % name)
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Multi-valued hyperparameters cannot be"
                                 " empty: %s" % name)
            self._hparam_types[name] = (type(value[0]), True)
        else:
            self._hparam_types[name] = (type(value), False)
        setattr(self, name, value)

    def parse(self, values):
        type_map = dict()

        for name, t in six.iteritems(self._hparam_types):
            param_type, _ = t
            type_map[name] = param_type

        values_map = parse_values(values, type_map)
        return self._set_from_map(values_map)

    def _set_from_map(self, values_map):
        for name, value in six.iteritems(values_map):
            if name not in self._hparam_types:
                logging.debug("%s not found in hparams." % name)
                continue

            _, is_list = self._hparam_types[name]

            if isinstance(value, list):
                if not is_list:
                    raise ValueError("Must not pass a list for single-valued "
                                     "parameter: %s" % name)
                setattr(self, name, value)
            else:
                if is_list:
                    raise ValueError("Must pass a list for multi-valued "
                                     "parameter: %s" % name)
                setattr(self, name, value)
        return self

    def to_json(self):
        return json.dumps(self.values())

    def parse_json(self, values_json):
        values_map = json.loads(values_json)
        return self._set_from_map(values_map)

    def values(self):
        return {n: getattr(self, n) for n in six.iterkeys(self._hparam_types)}

    def __str__(self):
        return str(sorted(six.iteritems(self.values)))
