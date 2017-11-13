#################################################
# Optimizers implemented based on recent Synchronous Distributed SGD design principles
#
# https://research.fb.com/publications/accurate-large-minibatch-sgd-training-imagenet-in-1-hour/
#################################################


import numpy as np

import tensorflow as tf

from learning_rates import tf_linear_scaling_lr


class AbstractRayOptimizer(object):

    def __init__(self, num_minibatches, num_workers, worker_minibatch_size):
        """
        A distributed optimizer implementation designed to operate on
        n examples for minibatch sizes of n*k, where
        n = worker_minibatch_size and k = num_workers.

        :param num_minibatches: Total number of minibatches.
        :param worker_minibatch_size: The number of examples for this model (per-worker batchsize).
        :param num_workers: The total number of workers.
        """
        self.worker_minibatch_size = worker_minibatch_size
        self.num_workers = num_workers
        self.num_minibatches = num_minibatches

    def compute_gradient(self, x, y):
        """
        Assume we have a gradient function of the loss with respect to weights, normalized by n.
        Compute the gradient for (x, y), and divide by k to normalize by minibatch size k*n (see remark 3).

        :param x: n input examples.
        :param y: n output examples.
        :return: The gradient of weights at (x, y) normalized by minibatch size.
        """
        raise NotImplementedError("Implement a way to compute the weight gradient.")

    def apply_gradients(self, *gradients):
        """
        Given a list of k minibatch normalized gradients, compute their sum and
        apply a gradient update using some optimizer. Implements optimizer corrections (see remarks 1 and 2).

        :param gradients: A list of k gradients.
        :return: Nothing.
        """
        raise NotImplementedError("Implement a way to apply a weight update with k gradients.")

    def compute_loss(self, x, y):
        """
        Compute loss.
        :param x: Input values for loss computation.
        :param y: Output values for loss computation.
        :return: The loss.
        """
        raise NotImplementedError("Implement computation of loss.")


class TFRayOptimizer(AbstractRayOptimizer):
    # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py#L306
    # and https://stackoverflow.com/questions/34910452/tensorflow-apply-gradients-remotely

    def __init__(self,
                 session,
                 model,
                 num_minibatches,
                 num_workers,
                 worker_minibatch_size,
                 opt,
                 global_step_var):
        super(TFRayOptimizer, self).__init__(num_minibatches, num_workers, worker_minibatch_size)

        self._session = session
        self._model = model
        self._grads_and_vars = None
        self._grad_placeholder = None
        self._apply_placeholder_op = None
        self._opt = opt

        self._build(global_step_var)

    def _build(self, global_step_var):
        # a list of (gradient, variable) pairs, where gradient may be None
        # this gives the variables in tf that the optimizer is going to use to compute the gradient
        self._grads_and_vars = self._opt.compute_gradients(self._model.loss)
        vars_with_grads = list(filter(lambda x: x[0] is not None, self._grads_and_vars))
        if not vars_with_grads:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in self._grads_and_vars], self._model.loss))
        # keep only grad/var pairs for vars that have grads.
        # these are the variables that have a gradient wrt the loss.
        # variables inspected aren't scoped to the computation graph of the loss.
        # e.g. if we have two models in a session, compute_gradients will
        # look at variables in both loss nodes.
        self._grads_and_vars = vars_with_grads
        # this creates a place holder for setting the gradient (cpu memory to gpu memory)
        self._grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in self._grads_and_vars]
        # the tuple in the list of tuples used here
        # pair the gradients from physical memory with variables in gpu memory.
        self._apply_placeholder_op = self._opt.apply_gradients(self._grad_placeholder, global_step=global_step_var)

    def compute_gradient(self, x, y):
        feed_dict = {self._model.x_holder: x, self._model.y_holder: y}
        grad_vals = self._session.run([grad[0] for grad in self._grads_and_vars], feed_dict=feed_dict)
        # TODO: move into tensorflow
        return list(map(lambda grad: grad/self.num_workers, grad_vals))

    def apply_gradients(self, *gradients):
        # this may not be worth doing in tensorflow if we implement reduce
        return self.apply_gradient(list(map(lambda x: np.sum(x, axis=0), zip(*gradients))))

    def apply_gradient(self, gradient):
        feed_dict = {}
        for i in range(len(self._grad_placeholder)):
            feed_dict[self._grad_placeholder[i][0]] = gradient[i]
        self._session.run(self._apply_placeholder_op, feed_dict=feed_dict)

    def compute_loss(self, x, y):
        feed_dict = {self._model.x_holder: x, self._model.y_holder: y}
        return self._session.run(self._model.loss, feed_dict=feed_dict)


class TFGDRayOptimizer(TFRayOptimizer):

    def __init__(self,
                 session,
                 model,
                 num_minibatches,
                 num_workers,
                 worker_minibatch_size,
                 init_learning_rate=0.01,
                 warmup_epochs=5):
        global_step_var, lr = tf_linear_scaling_lr(num_minibatches, num_workers, init_learning_rate, warmup_epochs)
        self.lr = lr
        opt = tf.train.GradientDescentOptimizer(lr)
        super(TFGDRayOptimizer, self).__init__(session, model,
                                               num_minibatches,
                                               num_workers,
                                               worker_minibatch_size,
                                               opt, global_step_var,)


if __name__ == "__main__":
    pass
