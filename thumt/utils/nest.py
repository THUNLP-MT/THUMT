# coding=utf-8
# Copyright 2017-2020 The THUMT Authors
# Modified from TensorFlow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six


def _sorted(dict_):
    try:
        return sorted(six.iterkeys(dict_))
    except TypeError:
        raise TypeError("nest only supports dicts with sortable keys.")


def _sequence_like(instance, args):
    if isinstance(instance, dict):
        result = dict(zip(_sorted(instance), args))
        return type(instance)((key, result[key])
                              for key in six.iterkeys(instance))
    elif (isinstance(instance, tuple) and
          hasattr(instance, "_fields") and
          isinstance(instance._fields, collections.Sequence) and
          all(isinstance(f, six.string_types) for f in instance._fields)):
        # This is a namedtuple
        return type(instance)(*args)
    else:
        # Not a namedtuple
        return type(instance)(args)


def _yield_value(iterable):
    if isinstance(iterable, dict):
        for key in _sorted(iterable):
            yield iterable[key]
    else:
        for value in iterable:
            yield value


def _yield_flat_nest(nest):
    for n in _yield_value(nest):
        if is_sequence(n):
            for ni in _yield_flat_nest(n):
                yield ni
        else:
            yield n


def is_sequence(seq):
    if isinstance(seq, dict):
        return True
    if isinstance(seq, set):
        print("Sets are not currently considered sequences, but this may "
              "change in the future, so consider avoiding using them.")
    return (isinstance(seq, collections.Sequence)
            and not isinstance(seq, six.string_types))


def flatten(nest):
    if is_sequence(nest):
        return list(_yield_flat_nest(nest))
    else:
        return [nest]


def _recursive_assert_same_structure(nest1, nest2, check_types):
    is_sequence_nest1 = is_sequence(nest1)
    if is_sequence_nest1 != is_sequence(nest2):
        raise ValueError(
            "The two structures don't have the same nested structure.\n\n"
            "First structure: %s\n\nSecond structure: %s." % (nest1, nest2))

    if not is_sequence_nest1:
        return  # finished checking

    if check_types:
        type_nest1 = type(nest1)
        type_nest2 = type(nest2)
        if type_nest1 != type_nest2:
            raise TypeError(
                "The two structures don't have the same sequence type. First "
                "structure has type %s, while second structure has type %s."
                % (type_nest1, type_nest2))

        if isinstance(nest1, dict):
            keys1 = set(_six.iterkeys(nest1))
            keys2 = set(_six.iterkeys(nest2))
            if keys1 != keys2:
                raise ValueError(
                    "The two dictionaries don't have the same set of keys. "
                    "First structure has keys {}, while second structure has"
                    " keys {}.".format(keys1, keys2))

    nest1_as_sequence = [n for n in _yield_value(nest1)]
    nest2_as_sequence = [n for n in _yield_value(nest2)]
    for n1, n2 in zip(nest1_as_sequence, nest2_as_sequence):
        _recursive_assert_same_structure(n1, n2, check_types)


def assert_same_structure(nest1, nest2, check_types=True):
    len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
    len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
    if len_nest1 != len_nest2:
        raise ValueError("The two structures don't have the same number of "
                         "elements.\n\nFirst structure (%i elements): %s\n\n"
                         "Second structure (%i elements): %s"
                         % (len_nest1, nest1, len_nest2, nest2))
    _recursive_assert_same_structure(nest1, nest2, check_types)


def flatten_dict_items(dictionary):
    if not isinstance(dictionary, dict):
        raise TypeError("input must be a dictionary")
    flat_dictionary = {}
    for i, v in six.iteritems(dictionary):
        if not is_sequence(i):
            if i in flat_dictionary:
                raise ValueError(
                    "Could not flatten dictionary: key %s is not unique." % i)
            flat_dictionary[i] = v
        else:
            flat_i = flatten(i)
            flat_v = flatten(v)
            if len(flat_i) != len(flat_v):
                raise ValueError(
                    "Could not flatten dictionary. Key had %d elements, but"
                    " value had %d elements. Key: %s, value: %s."
                    % (len(flat_i), len(flat_v), flat_i, flat_v))
            for new_i, new_v in zip(flat_i, flat_v):
                if new_i in flat_dictionary:
                    raise ValueError(
                        "Could not flatten dictionary: key %s is not unique."
                        % (new_i))
                flat_dictionary[new_i] = new_v
    return flat_dictionary


def _packed_nest_with_indices(structure, flat, index):
    packed = []
    for s in _yield_value(structure):
        if is_sequence(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(_sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def pack_sequence_as(structure, flat_sequence):
    if not is_sequence(flat_sequence):
        raise TypeError("flat_sequence must be a sequence")

    if not is_sequence(structure):
        if len(flat_sequence) != 1:
            raise ValueError("Structure is a scalar but len(flat_sequence) =="
                             " %d > 1" % len(flat_sequence))
        return flat_sequence[0]

    flat_structure = flatten(structure)
    if len(flat_structure) != len(flat_sequence):
        raise ValueError(
            "Could not pack sequence. Structure had %d elements, but "
            "flat_sequence had %d elements.  Structure: %s, flat_sequence: %s."
            % (len(flat_structure), len(flat_sequence), structure,
               flat_sequence))

    _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
    return _sequence_like(structure, packed)


def map_structure(func, *structure, **check_types_dict):
    if not callable(func):
        raise TypeError("func must be callable, got: %s" % func)

    if not structure:
        raise ValueError("Must provide at least one structure")

    if check_types_dict:
        if "check_types" not in check_types_dict or len(check_types_dict) > 1:
            raise ValueError("Only valid keyword argument is check_types")
        check_types = check_types_dict["check_types"]
    else:
        check_types = True

    for other in structure[1:]:
        assert_same_structure(structure[0], other, check_types=check_types)

    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)

    return pack_sequence_as(
        structure[0], [func(*x) for x in entries])
