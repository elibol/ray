#################################################
# Ray implementation of synchronous distributed sgd
# with Tensorflow backend.
#
# Minimize server role in case we decide to factor
# it out in favor of a decentralized implementation
# (e.g. some variant of all_reduce on workers).
#################################################


import os
import math

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import ray

from models import SimpleModel, LinearTFRayModel
from optimizers import TFGDRayOptimizer
from utils import get_batches, get_worker_batch, arg_shuffle


class SyncSGDActor(object):

    # TODO: can we call super on actors that override __init__?
    def _init(self,
              name,
              train_size,
              num_workers,
              worker_minibatch_size,
              init_eta,
              warmup_epochs,
              init_seed):

        N = self._train_size = train_size
        n = self._worker_minibatch_size = worker_minibatch_size
        k = self._num_workers = num_workers
        M = self._num_minibatches = num_minibatches = int(math.ceil(N / (n * k)))

        # init session, model, and optimizer
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        with tf.device("/cpu:0"):
            # restrict to a single cpu for testing
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1)
            self._session = tf.Session(config=session_conf)
        with tf.name_scope(name) as scope:
            self._model = LinearTFRayModel(self._session, shape=(784, 10))
            self._opt = TFGDRayOptimizer(self._session,
                                         self._model,
                                         num_minibatches,
                                         num_workers,
                                         worker_minibatch_size,
                                         init_learning_rate=init_eta,
                                         warmup_epochs=warmup_epochs)
        self._session.run(tf.initialize_all_variables())
        tf.summary.FileWriter('./tensorboard/%s' % name, self._session.graph)

        self._init_seed = init_seed
        self._train, self._val, self._test = self._fetch_data()

        self.name = name

    def _fetch_data(self):
        data = input_data.read_data_sets("MNIST_data", one_hot=True, seed=self._init_seed)
        train = (data.train.images, data.train.labels)
        val = data.validation.images, data.validation.labels
        test = data.test.images, data.test.labels
        return train, val, test


@ray.remote
class SyncSGDWorker(SyncSGDActor):

    def __init__(self,
                 train_size,
                 num_workers,
                 worker_minibatch_size,
                 init_eta,
                 warmup_epochs,
                 init_seed,
                 worker_index):

        name = 'worker_%d' % worker_index
        self._init(
            name, train_size, num_workers, worker_minibatch_size,
            init_eta, warmup_epochs, init_seed
        )

        self._server = None
        self._worker_index = worker_index
        self._gradient = None
        self._idx = None

    def set_server(self, server):
        self._server = server

    def receive_weights(self, weights):
        self._model.weights = weights

    def shuffle(self, epoch):
        shuffle_seed = self._init_seed + epoch
        self._idx = arg_shuffle(self._train[0], shuffle_seed)

    def compute_gradient(self, minibatch_index):
        N = self._train_size
        n = self._worker_minibatch_size
        k = self._num_workers
        i = self._worker_index
        j = minibatch_index
        s, e = get_worker_batch(N, n, k, i, j)
        idx = self._idx
        x_batch, y_batch = self._train[0][idx][s:e], self._train[1][idx][s:e]
        return self._opt.compute_gradient(x_batch, y_batch)

    def send_gradient(self, gradient):
        self._server.receive_gradient.remote(gradient, self._worker_index)

    def compute_and_send(self, minibatch_index):
        gradient = self.compute_gradient(minibatch_index)
        self.send_gradient(gradient=gradient)
        return True

    def weights(self):
        return self._model.weights


@ray.remote
class SyncSGDServer(SyncSGDActor):

    def __init__(self,
                 train_size,
                 num_workers,
                 worker_minibatch_size,
                 init_eta,
                 warmup_epochs,
                 init_seed):

        name = 'server'
        self._init(
            name, train_size, num_workers, worker_minibatch_size,
            init_eta, warmup_epochs, init_seed
        )

        self._workers = [None]*num_workers
        self._gradients = [None]*num_workers

        self._init_weights()

    def _init_weights(self):
        rs = np.random.RandomState(seed=self._init_seed)
        self._model.weights = 1e-2 * rs.normal(size=self._model.num_weights)

    def set_worker(self, worker, worker_index):
        self._workers[worker_index] = worker

    def broadcast_weights(self):
        weights = self._model.weights
        ray.get([worker.receive_weights.remote(weights) for worker in self._workers])

    def receive_gradient(self, gradient, worker_index):
        self._gradients[worker_index] = gradient

    def apply_stored_gradients(self):
        self.apply_gradients(self._gradients)

    def apply_gradients(self, gradients):
        self._opt.apply_gradients(*gradients)

    def apply_and_broadcast(self, *gradients):
        self._opt.apply_gradients(*gradients)
        self.broadcast_weights()

    def loss(self):
        return self._opt.compute_loss(self._val[0], self._val[1])

    def weights(self):
        return self._model.weights


def main():
    ray.init(redirect_output=True)
    seed = 1337

    N = train_size = input_data.read_data_sets("MNIST_data", one_hot=True, seed=seed).train.images.shape[0]
    n = worker_minibatch_size = 32
    k = num_workers = 6
    num_minibatches = int(math.ceil(N / (n * k)))

    # scale from this to init_eta * k after warmup_epochs
    init_eta = 0.1
    warmup_epochs = 1

    total_epochs = 10

    # init actors
    server = SyncSGDServer.remote(train_size, num_workers, worker_minibatch_size, init_eta, warmup_epochs, seed)
    workers = [None]*k
    for i in range(k):
        # TODO: issue passing multiple actors as a list.
        workers[i] = SyncSGDWorker.remote(
            train_size, num_workers, worker_minibatch_size,
            init_eta, warmup_epochs, seed, i
        )
        ray.get(workers[i].set_server.remote(server))
        ray.get(server.set_worker.remote(workers[i], i))

    # train
    for epoch in range(total_epochs):
        for i in range(num_workers):
            ray.get(workers[i].shuffle.remote(epoch))
        for minibatch_index in range(num_minibatches):
            results = [worker.compute_gradient.remote(minibatch_index) for worker in workers]
            ray.get(server.apply_and_broadcast.remote(*results))
            print(epoch, "%.3d" % minibatch_index, ray.get(server.loss.remote()))


if __name__ == "__main__":
    main()
