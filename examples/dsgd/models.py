#################################################
# Simple model abstractions with TF imps.
#################################################

import numpy as np
import tensorflow as tf

from ray.experimental.tfutils import TensorFlowVariables


class AbstractRayModel(object):

    def __init__(self):
        """
        A model interface.
        """
        pass

    @property
    def weights(self):
        """
        :return: A flat vector of weights.
        """
        raise NotImplementedError("Implement weight getter.")

    @weights.setter
    def weights(self, weights):
        """
        :param weights: A flat vector of weights.
        :return: None
        """
        raise NotImplementedError("Implement weight setter.")


class TFRayModel(AbstractRayModel):

    def __init__(self, session, shape):
        super(TFRayModel, self).__init__()

        self._session = session
        x_holder, y_holder, loss = self._build(shape)
        self._variables = TensorFlowVariables(loss, self._session)

        self.x_holder, self.y_holder, self.loss = x_holder, y_holder, loss
        self.num_weights = int(self._variables.get_flat_size())

    def _build(self, shape):
        raise NotImplemented("Missing model implementation.")

    @property
    def weights(self):
        return self._variables.get_flat()

    @weights.setter
    def weights(self, weights):
        self._variables.set_flat(weights)


class LinearTFRayModel(TFRayModel):

    def _build(self, shape):
        x = tf.placeholder(tf.float32, [None, shape[0]])
        w = tf.Variable(tf.zeros(shape))
        b = tf.Variable(tf.zeros(shape[1]))
        y = tf.nn.softmax(tf.matmul(x, w) + b)
        y_ = tf.placeholder(tf.float32, [None, shape[1]])
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        return x, y_, loss


class SimpleModel(TFRayModel):

    def _build(self, shape):
        x = tf.placeholder("float", None)
        y_ = tf.placeholder("float", None)
        w = tf.Variable(0.0)
        loss = tf.abs(y_ - x * w ** 2)
        return x, y_, loss
