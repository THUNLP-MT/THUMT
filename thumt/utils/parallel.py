# coding=utf-8
# Copyright 2017 The THUMT Authors

import six
import tensorflow as tf


def _maybe_repeat(x, n):
    if isinstance(x, list):
        assert len(x) == n
        return x
    else:
        return [x] * n


def getter_fn(cache):
    def daisy_chain_getter(getter, name, *args, **kwargs):
        device_var_key = (devices[i], name)
        if device_var_key in cache:
            # If we have the variable on the correct device, return it.
            return cache[device_var_key]
        if name in cache:
            # If we have it on a different device, copy it from the last
            # device
            v = tf.identity(cache[name])
        else:
            var = getter(name, *args, **kwargs)
            v = tf.identity(var._ref())
        # Update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

    return daisy_chain_getter


# Data-level parallelism
def data_parallelism(devices, fn, *args, **kwargs):
    num_worker = len(devices)

    # Replicate args and kwargs
    if args:
        new_args = [_maybe_repeat(arg, num_worker) for arg in args]
        # Transpose
        new_args = [list(x) for x in zip(*new_args)]
    else:
        new_args = [[] for _ in xrange(num_worker)]

    new_kwargs = [{} for _ in xrange(num_worker)]

    for k, v in six.iteritems(kwargs):
        vals = _maybe_repeat(v, num_worker)

        for i in xrange(num_worker):
            new_kwargs[i][k] = vals[i]

    fns = _maybe_repeat(fn, num_worker)

    # Now make the parallel call.
    outputs = []
    cache = {}

    for i in xrange(num_worker):
        with tf.name_scope('parallel_%d' % i):
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if i > 0 else None,
                                   custom_getter=getter_fn(cache)):
                with tf.device(devices[i]):
                    outputs.append(fns[i](*new_args[i], **new_kwargs[i]))

    if isinstance(outputs[0], tuple):
        outputs = list(zip(*outputs))
        outputs = tuple([list(o) for o in outputs])

    return outputs


def shard_features(features, device_list):
    num_datashards = len(device_list)

    sharded_features = {}

    for k, v in features.iteritems():
        v = tf.convert_to_tensor(v)
        if not v.shape.as_list():
            v = tf.expand_dims(v, axis=-1)
            v = tf.tile(v, [num_datashards])
        with tf.device(v.device):
            sharded_features[k] = tf.split(v, num_datashards, 0)

    datashard_to_features = []

    for d in xrange(num_datashards):
        feat = {
            k: v[d] for k, v in six.iteritems(sharded_features)
        }
        datashard_to_features.append(feat)

    return datashard_to_features


def parallel_model(model_fn, features, devices, use_cpu=False):
    devices = ["gpu:%d" % d for d in devices]

    if use_cpu:
        devices += ["cpu:0"]

    if len(devices) == 1:
        return [model_fn(features)]

    features = shard_features(features, devices)

    outputs = data_parallelism(devices, model_fn, features)
    return outputs
