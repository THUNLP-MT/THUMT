# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import tensorflow as tf


class GPUParamServerDeviceSetter(object):

    def __init__(self, worker_device, ps_devices):
        self.ps_devices = ps_devices
        self.worker_device = worker_device
        self.ps_sizes = [0] * len(self.ps_devices)

    def __call__(self, op):
        if op.device:
            return op.device
        if op.type not in ["Variable", "VariableV2", "VarHandleOp"]:
            return self.worker_device

        # Gets the least loaded ps_device
        device_index, _ = min(enumerate(self.ps_sizes),
                              key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size

        return device_name


def _maybe_repeat(x, n):
    if isinstance(x, list):
        assert len(x) == n
        return x
    else:
        return [x] * n


def _create_device_setter(is_cpu_ps, worker, num_gpus):
    if is_cpu_ps:
        # tf.train.replica_device_setter supports placing variables on the CPU,
        # all on one GPU, or on ps_servers defined in a cluster_spec.
        return tf.train.replica_device_setter(
            worker_device=worker, ps_device="/cpu:0", ps_tasks=1)
    else:
        gpus = ["/gpu:%d" % i for i in range(num_gpus)]
        return GPUParamServerDeviceSetter(worker, gpus)


# Data-level parallelism
def data_parallelism(devices, fn, *args, **kwargs):
    num_worker = len(devices)

    # Replicate args and kwargs
    if args:
        new_args = [_maybe_repeat(arg, num_worker) for arg in args]
        # Transpose
        new_args = [list(x) for x in zip(*new_args)]
    else:
        new_args = [[] for _ in range(num_worker)]

    new_kwargs = [{} for _ in range(num_worker)]

    for k, v in kwargs.iteritems():
        vals = _maybe_repeat(v, num_worker)

        for i in range(num_worker):
            new_kwargs[i][k] = vals[i]

    fns = _maybe_repeat(fn, num_worker)

    # Now make the parallel call.
    outputs = []

    for i in range(num_worker):
        worker = "/gpu:%d" % i
        device_setter = _create_device_setter(False, worker, len(devices))
        with tf.variable_scope(tf.get_variable_scope(), reuse=(i != 0)):
            with tf.name_scope("parallel_%d" % i):
                with tf.device(device_setter):
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

    for d in range(num_datashards):
        feat = {
            k: v[d] for k, v in sharded_features.iteritems()
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
