# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from thumt.utils.distribute import all_reduce


class LossScalingOptimizer(tf.train.Optimizer):

    def __init__(self, optimizer, scale=128.0, use_locking=False,
                 name="LossScalingOptimizer"):
        super(LossScalingOptimizer, self).__init__(use_locking, name)
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


class MultiStepOptimizer(tf.train.Optimizer):

    def __init__(self, optimizer, step=1, use_locking=False,
                 name="MultiStepOptimizer"):
        super(MultiStepOptimizer, self).__init__(use_locking, name)
        self._optimizer = optimizer
        self._step = step

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

        step_t = tf.convert_to_tensor(self._step, name="step")

        # Create slots for gradient accumulators
        for v in var_list:
            self._zeros_slot(v, "grad_acc", self._name)

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
            iter_op = iter_var.assign(tf.mod(iter_var + 1, step_t),
                                          use_locking=self._use_locking)

        return tf.group(*[update_op, iter_op])
