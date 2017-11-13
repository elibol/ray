#################################################
# Learning rates implemented based on recent Synchronous Distributed SGD design principles
#
# https://research.fb.com/publications/accurate-large-minibatch-sgd-training-imagenet-in-1-hour/
#################################################


import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def linear_scaling_lr(init_eta, k, epoch):
    # gradual warmup implementation (see 2.2)
    warmup_period = 5.0
    eta = init_eta * k
    if epoch < warmup_period:
        delta = (eta - init_eta) * epoch / warmup_period
        return init_eta + delta
    else:
        return eta


def tf_linear_scaling_lr(num_minibatches, num_workers, init_eta, warmup_epochs=5):
    # gradual warmup implementation (section 2.2)
    global_step = tf.Variable(0, trainable=False)
    warmup_steps = num_minibatches * warmup_epochs
    return global_step, tf_linear_warmup(init_eta, init_eta * num_workers, warmup_steps, global_step)


def tf_linear_warmup(start_learning_rate, end_learning_rate, warmup_steps, global_step_var, name=None):
    # adapted from
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/learning_rate_decay.py#L182

    global_step = global_step_var
    if global_step is None:
        raise ValueError("global_step is required.")
    with ops.name_scope(name, "LearningRateWarmup",
                        [start_learning_rate, global_step, warmup_steps, end_learning_rate]) as name:
        start_learning_rate = ops.convert_to_tensor(start_learning_rate, name="learning_rate")
        dtype = start_learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        warmup_steps = math_ops.cast(warmup_steps, dtype)
        end_learning_rate = math_ops.cast(end_learning_rate, dtype)
        # make sure that the global_step used is not bigger than decay_steps
        global_step = math_ops.minimum(global_step, warmup_steps)
        p = math_ops.div(global_step, warmup_steps)
        return math_ops.subtract(end_learning_rate,
                                 math_ops.multiply(end_learning_rate - start_learning_rate, 1 - p),
                                 name=name)

