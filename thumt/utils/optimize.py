# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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


def create_train_op(loss, optimizer, global_step, params):

    with tf.name_scope("create_train_op"):
        grads_and_vars = optimizer.compute_gradients(
            loss, colocate_gradients_with_ops=True)
        gradients = [item[0] for item in grads_and_vars]
        variables = [item[1] for item in grads_and_vars]

        if params.update_cycle == 1:
            zero_variables_op = tf.no_op("zero_variables")
            collect_op = tf.no_op("collect_op")
        else:
            loss_var = tf.Variable(tf.zeros([]),  name="loss/replica",
                                   trainable=False)
            slot_variables = _replicate_variables(variables)
            zero_variables_op = _zero_variables(slot_variables + [loss_var])
            collect_grads_op = _collect_gradients(gradients, slot_variables)
            collect_loss_op = tf.assign_add(loss_var, loss)
            collect_op = tf.group(collect_loss_op, collect_grads_op,
                                  name="collect_op")
            scale = 1.0 / params.update_cycle
            gradients = [scale * (g + s)
                         for (g, s) in zip(gradients, slot_variables)]
            loss = scale * (loss + loss_var)

        global_norm = tf.global_norm(gradients)
        tf.summary.scalar("global_norm/gradient_norm", global_norm)

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
                                                  params.clip_grad_norm,
                                                  use_norm=global_norm)

        # Update variables
        grads_and_vars = list(zip(gradients, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        ops = {
            "zero_op": zero_variables_op,
            "collect_op": collect_op,
            "train_op": train_op
        }

    # Add summaries
    tf.summary.scalar(params.model + "/loss", loss)

    return loss, ops
