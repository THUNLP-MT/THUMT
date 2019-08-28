# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from thumt.utils.distribute import all_reduce


class StaticLossScalingOptimizer(tf.train.Optimizer):

    def __init__(self, optimizer, scale=128.0, use_locking=False,
                 name="StaticLossScalingOptimizer"):
        super(StaticLossScalingOptimizer, self).__init__(use_locking, name)
        self._optimizer = optimizer
        self._scale = scale

    def compute_gradients(self, loss, var_list=None,
                          gate_gradients=tf.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        grads_and_vars = self._optimizer.compute_gradients(
            loss * self._scale, var_list, gate_gradients,
            aggregation_method, colocate_gradients_with_ops, grad_loss)

        scaled_grads_and_vars = []

        for grad, var in grads_and_vars:
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.IndexedSlices(grad.values / self._scale,
                                        grad.indices,  grad.dense_shape)
            elif isinstance(grad, tf.Tensor):
                grad = grad / self._scale

            scaled_grads_and_vars.append((grad, var))

        return scaled_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        return self._optimizer.apply_gradients(grads_and_vars, global_step,
                                               name)


class LossScalingOptimizer(tf.train.Optimizer):

    def __init__(self, optimizer, scale=2.0**15, scale_factor=2.0,
                 scale_window=2000, tolerance=0.05, threshold=None,
                 use_locking=False, name="LossScalingOptimizer"):
        super(LossScalingOptimizer, self).__init__(use_locking, name)
        self._optimizer = optimizer
        self._scale = tf.convert_to_tensor(scale, dtype=tf.float32)
        self._scale_factor = tf.convert_to_tensor(scale_factor,
                                                  dtype=tf.float32)
        self._scale_window = tf.convert_to_tensor(scale_window,
                                                  dtype=tf.int32)
        self._tolerance = tf.convert_to_tensor(tolerance,
                                               dtype=tf.float32)

        if threshold:
            self._threshold = tf.convert_to_tensor(threshold,
                                                   dtype=tf.float32)
        else:
            self._threshold = None

    def compute_gradients(self, loss, var_list=None,
                          gate_gradients=tf.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        scale_var = self._create_non_slot_variable(
            initial_value=self._scale, name="scale", colocate_with=loss)

        grads_and_vars = self._optimizer.compute_gradients(
            loss * scale_var, var_list, gate_gradients,
            aggregation_method, colocate_gradients_with_ops, grad_loss)

        scaled_grads_and_vars = []

        for grad, var in grads_and_vars:
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.IndexedSlices(grad.values / scale_var,
                                        grad.indices,  grad.dense_shape)
            elif isinstance(grad, tf.Tensor):
                grad = grad / scale_var

            scaled_grads_and_vars.append((grad, var))

        return scaled_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, var_list = list(zip(*grads_and_vars))
        grad_norm = tf.global_norm(grads)
        new_grads = []

        is_overflow = tf.logical_not(tf.is_finite(grad_norm))

        for grad in grads:
            if grad is not None:
                grad = tf.cond(is_overflow,
                               lambda: tf.zeros_like(grad), lambda: grad)
            new_grads.append(grad)

        grads_and_vars = list(zip(new_grads, var_list))
        update_op = self._optimizer.apply_gradients(grads_and_vars,
                                                    global_step, name)

        first_var = min(var_list, key=lambda x: x.name)
        iter_var = self._create_non_slot_variable(
            initial_value=0, name="iter", colocate_with=first_var)
        rescale_iter = self._create_non_slot_variable(
            initial_value=0, name="rescale_iter", colocate_with=first_var)
        overflow_iter = self._create_non_slot_variable(
            initial_value=-1, name="overflow_iter", colocate_with=first_var)
        overflow_var = self._create_non_slot_variable(
            initial_value=0, name="overflow", colocate_with=first_var)
        scale_var = self._get_non_slot_variable("scale",
                                                tf.get_default_graph())

        iter_op = tf.assign_add(iter_var, 1)
        overflow_count_op = tf.cond(
            is_overflow,
            lambda: tf.assign_add(overflow_var, 1),
            lambda: overflow_var)
        overflow_iter_op = tf.cond(
            is_overflow,
            lambda: tf.assign(overflow_iter, iter_var),
            lambda: iter_var)

        def increase_scale():
            scale_op = tf.assign(scale_var, scale_var * self._scale_factor,
                                 use_locking=self._use_locking)
            iter_op = tf.assign(rescale_iter, iter_var,
                                use_locking=self._use_locking)
            return tf.group(*[scale_op, iter_op])

        def decrease_scale():
            scale_op = tf.assign(scale_var, scale_var / self._scale_factor,
                                 use_locking=self._use_locking)
            if self._threshold is not None:
                scale_op = tf.assign(scale_op, tf.maximum(scale_var,
                                                          self._threshold))

            iter_op = tf.assign(rescale_iter, iter_var,
                                use_locking=self._use_locking)
            overflow_op = tf.assign(overflow_var, 0)

            return tf.group(*[scale_op, iter_op, overflow_op])

        with tf.control_dependencies([overflow_count_op, overflow_iter_op]):
            percentage = tf.div(tf.cast(overflow_var, tf.float32),
                                tf.cast(iter_var - rescale_iter, tf.float32))
            decrease_scale_op = tf.cond(
                tf.logical_and(is_overflow,
                               tf.greater(percentage, self._tolerance)),
                decrease_scale, lambda: tf.no_op())
            increase_scale_op = tf.cond(
                tf.logical_and(
                    tf.logical_not(is_overflow),
                    tf.equal(
                        (iter_var - overflow_iter) % self._scale_window, 0)
                ), increase_scale, lambda: tf.no_op())

        ops = [
            update_op, iter_op, overflow_count_op, overflow_iter_op,
            increase_scale_op, decrease_scale_op
        ]

        return tf.group(*ops)


class MultiStepOptimizer(tf.train.Optimizer):

    def __init__(self, optimizer, step=1, use_locking=False,
                 name="MultiStepOptimizer"):
        super(MultiStepOptimizer, self).__init__(use_locking, name)
        self._optimizer = optimizer
        self._step = step
        self._step_t = tf.convert_to_tensor(step, name="step")

    def _all_reduce(self, tensor):
        with tf.name_scope(self._name + "_Allreduce"):
            if tensor is None:
                return tensor

            if isinstance(tensor, tf.IndexedSlices):
                tensor = tf.convert_to_tensor(tensor)

            return all_reduce(tensor)

    def compute_gradients(self, loss, var_list=None,
                          gate_gradients=tf.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        grads_and_vars = self._optimizer.compute_gradients(loss , var_list,
            gate_gradients, aggregation_method, colocate_gradients_with_ops,
            grad_loss)

        grads, var_list = list(zip(*grads_and_vars))

        # Do not create extra variables when step is 1
        if self._step == 1:
            grads = [self._all_reduce(t) for t in grads]
            return list(zip(grads, var_list))

        first_var = min(var_list, key=lambda x: x.name)
        iter_var = self._create_non_slot_variable(
            initial_value=0 if self._step == 1 else 1, name="iter",
            colocate_with=first_var)

        new_grads = []

        for grad, var in zip(grads, var_list):
            grad_acc = self._zeros_slot(var, "grad_acc", self._name)

            if isinstance(grad, tf.IndexedSlices):
                grad_acc = tf.scatter_add(grad_acc, grad.indices, grad.values,
                                          use_locking=self._use_locking)
            else:
                grad_acc = tf.assign_add(grad_acc, grad,
                                         use_locking=self._use_locking)

            def _acc_grad():
                return grad_acc

            def _avg_grad():
                return self._all_reduce(grad_acc / self._step)

            grad = tf.cond(tf.equal(iter_var, 0), _avg_grad, _acc_grad)
            new_grads.append(grad)

        return list(zip(new_grads, var_list))

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if self._step == 1:
            return self._optimizer.apply_gradients(grads_and_vars, global_step,
                                                   name=name)

        grads, var_list = list(zip(*grads_and_vars))

        def _pass_gradients():
            return tf.group(*grads)

        def _apply_gradients():
            op = self._optimizer.apply_gradients(zip(grads, var_list),
                                                 global_step, name)
            with tf.control_dependencies([op]):
                zero_ops = []
                for var in var_list:
                    grad_acc = self.get_slot(var, "grad_acc")
                    zero_ops.append(
                        grad_acc.assign(tf.zeros_like(grad_acc),
                                        use_locking=self._use_locking))
                zero_op = tf.group(*zero_ops)
            return tf.group(*[op, zero_op])

        iter_var = self._get_non_slot_variable("iter", tf.get_default_graph())
        update_op = tf.cond(tf.equal(iter_var, 0), _apply_gradients,
                            _pass_gradients)

        with tf.control_dependencies([update_op]):
            iter_op = iter_var.assign(tf.mod(iter_var + 1, self._step_t),
                                      use_locking=self._use_locking)

        return tf.group(*[update_op, iter_op])
