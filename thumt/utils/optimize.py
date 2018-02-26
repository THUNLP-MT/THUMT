# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _get_loss_variable(graph=None):
    graph = graph or tf.get_default_graph()
    loss_tensors = tf.get_collection("loss")

    if len(loss_tensors) == 1:
        loss_tensor = loss_tensors[0]
    elif not loss_tensors:
        try:
            loss_tensor = graph.get_tensor_by_name("loss_tensor:0")
        except KeyError:
            return None
    else:
        tf.logging.error("Multiple tensors in loss collection.")
        return None

    return loss_tensor


def _create_loss_variable(graph=None):
    graph = graph or tf.get_default_graph()
    if _get_loss_variable(graph) is not None:
        raise ValueError("'loss' already exists.")

    # Create in proper graph and base name_scope.
    with graph.as_default() as g, g.name_scope(None):
        tensor = tf.get_variable("loss", shape=[], dtype=tf.float32,
                                 initializer=tf.zeros_initializer(),
                                 trainable=False,
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                              "loss"])

    return tensor


def _get_or_create_loss_variable(graph=None):
    graph = graph or tf.get_default_graph()
    loss_tensor = _get_loss_variable(graph)
    if loss_tensor is None:
        loss_tensor = _create_loss_variable(graph)
    return loss_tensor


def _zero_variables(variables, name=None):
    ops = []

    for var in variables:
        with tf.device(var.device):
            op = var.assign(tf.zeros(var.shape.as_list()))
        ops.append(op)

    return tf.group(*ops, name=name or "zero_variables")


def _replicate_variables(variables, device=None):
    new_vars = []

    for var in variables:
        device = device or var.device
        with tf.device(device):
            name = var.name.split(":")[0].rstrip("/") + "/replica"
            new_vars.append(tf.Variable(tf.zeros(var.shape.as_list()),
                                        name=name, trainable=False))

    return new_vars


def _collect_gradients(gradients, variables):
    ops = []

    for grad, var in zip(gradients, variables):
        if isinstance(grad, tf.Tensor):
            ops.append(tf.assign_add(var, grad))
        else:
            ops.append(tf.scatter_add(var, grad.indices, grad.values))

    return tf.group(*ops, name="collect_gradients")


def _scale_variables(variables, scale):
    if not isinstance(variables, (list, tuple)):
        return tf.assign(variables, scale * variables)

    ops = []

    for var in variables:
        ops.append(tf.assign(var, scale * var))

    return tf.group(*ops, name="scale_variables")


def create_train_op(loss, optimizer, global_step, params):
    with tf.name_scope("create_train_op"):
        grads_and_vars = optimizer.compute_gradients(
            loss, colocate_gradients_with_ops=True)
        gradients = [item[0] for item in grads_and_vars]
        variables = [item[1] for item in grads_and_vars]

        if params.update_cycle == 1:
            zero_variables_op = tf.no_op("zero_variables")
            collect_op = tf.no_op("collect_op")
            scale_op = tf.no_op("scale_op")
        else:
            # collect
            loss_tensor = _get_or_create_loss_variable()
            slot_variables = _replicate_variables(variables)
            zero_variables_op = _zero_variables(slot_variables + [loss_tensor])
            collect_grads_op = _collect_gradients(gradients, slot_variables)
            collect_loss_op = tf.assign_add(loss_tensor, loss)
            collect_op = tf.group(collect_loss_op, collect_grads_op,
                                  name="collect_op")
            # scale
            scale = 1.0 / params.update_cycle
            scale_grads_op = _scale_variables(slot_variables, scale)
            scale_loss_op = _scale_variables(loss_tensor, scale)
            scale_op = tf.group(scale_grads_op, scale_loss_op, name="scale_op")
            gradients = slot_variables
            loss = tf.convert_to_tensor(loss_tensor)

        # Add summaries
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_norm/gradient_norm",
                          tf.global_norm(gradients))

        for gradient, variable in zip(gradients, variables):
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient

            if grad_values is not None:
                var_name = variable.name.replace(":", "_")
                tf.summary.histogram("gradients/%s" % var_name, grad_values)
                tf.summary.scalar("gradient_norm/%s" % var_name,
                                  tf.global_norm([grad_values]))

        # Gradient clipping
        if isinstance(params.clip_grad_norm or None, float):
            gradients, _ = tf.clip_by_global_norm(gradients,
                                                  params.clip_grad_norm)

        # Update variables
        grads_and_vars = list(zip(gradients, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        ops = {
            "zero_op": zero_variables_op,
            "collect_op": collect_op,
            "scale_op": scale_op,
            "train_op": train_op
        }

        return loss, ops
