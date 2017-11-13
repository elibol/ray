#################################################
# Initial tests to be used/adapted to test suite
#################################################

import math

import numpy as np
import tensorflow as tf

from models import SimpleModel, LinearTFRayModel
from learning_rates import tf_linear_scaling_lr
from optimizers import TFGDRayOptimizer
from utils import get_batches, get_worker_batch, arg_shuffle


def test_batching(N=55000, n=32, k=4):
    M = math.ceil(N/(n*k))

    # this is correct, test against this.
    minibatches, worker_batches, last_worker_batches = get_batches(N, n, k)
    assert M == minibatches.num_batches

    for j in range(M):
        for i in range(k):
            computed_s, computed_e = get_worker_batch(N, n, k, i, j)
            # print("computed", j, computed_s, computed_e)

            minibatch = minibatches.batches[j]
            if j != M-1:
                workerbatch = worker_batches.batches[i]
            else:
                workerbatch = last_worker_batches.batches[i]

            stored_s, stored_e = minibatch[0] + workerbatch[0], minibatch[0] + workerbatch[1]
            # print("stored  ", j, stored_s, stored_e)
            assert computed_s == stored_s
            assert computed_e == stored_e


def test_opt_simple():
    init_weight = 3.0

    # scaling parameters
    num_minibatches = 3
    num_workers = 8
    worker_minibatch_size = 10
    init_eta = 0.1
    warmup_epochs = 5

    session = tf.Session()

    model = SimpleModel(session, shape=(1, 1))

    opt = TFGDRayOptimizer(session, model,
                           num_minibatches,
                           num_workers,
                           worker_minibatch_size,
                           init_learning_rate=init_eta,
                           warmup_epochs=warmup_epochs)

    session.run(tf.initialize_all_variables())
    model.weights = np.array([init_weight])

    tf.summary.FileWriter('./tensorboard/simplemodel', session.graph)

    w_1 = model.weights
    grad = opt.compute_gradient([3.0], [0.0])
    curr_lr = opt._session.run(opt.lr)
    opt.apply_gradient(grad)
    w_2 = model.weights

    assert np.allclose(w_1, init_weight)
    assert np.allclose(grad, 18 / num_workers)
    assert np.allclose(w_2, init_weight - 18 / num_workers * init_eta)
    assert np.allclose(curr_lr, 0.1)
    model.weights = np.array([init_weight])
    w_3 = model.weights
    assert np.allclose(w_1, w_3)
    # that's one step.

    # take warmup_steps - 1 steps.
    for i in range(num_minibatches * warmup_epochs - 1):
        curr_lr = opt._session.run(opt.lr)
        grad = opt.compute_gradient(3.0, 0.0)
        opt.apply_gradient(grad)
        assert curr_lr < init_eta * num_workers

    for i in range(100):
        curr_lr = opt._session.run(opt.lr)
        grad = opt.compute_gradient(3.0, 0.0)
        opt.apply_gradient(grad)
        assert np.allclose(curr_lr, init_eta * num_workers)


def test_linear_model():
    # TODO: make this deterministic
    from tensorflow.examples.tutorials.mnist import input_data

    def get_data():
        # init data and minibatch parameters
        data = input_data.read_data_sets("MNIST_data", one_hot=True, seed=1337)
        train = (data.train.images, data.train.labels)
        val = data.validation.images, data.validation.labels
        test = data.test.images, data.test.labels
        return train, val, test

    train, val, test = get_data()
    N = len(train[0])
    n = worker_minibatch_size = 32
    k = num_workers = 4
    M = num_minibatches = int(math.ceil(N / (n * k)))

    # init session
    session = tf.Session()

    # init models
    model = LinearTFRayModel(session, shape=(784, 10))

    # init optimizers
    init_eta = 0.1
    warmup_epochs = 5
    opt = TFGDRayOptimizer(session,
                           model,
                           num_minibatches,
                           num_workers,
                           worker_minibatch_size,
                           init_learning_rate=init_eta,
                           warmup_epochs=warmup_epochs)

    # init variables
    session.run(tf.initialize_all_variables())

    model.weights = 1e-2 * np.random.normal(size=model.num_weights)

    tf.summary.FileWriter('./tensorboard/linearmodel', session.graph)

    total_epochs = 10
    seed = 0
    prev_loss = None
    check_loss_every = 50
    num_checks = 0
    for epoch in range(total_epochs):
        seed += 1
        idx = arg_shuffle(train[0], seed)

        grads = [None] * num_workers
        for j in range(num_minibatches):
            for i in range(num_workers):
                s, e = get_worker_batch(N, n, k, i, j)
                x_batch, y_batch = train[0][idx][s:e], train[1][idx][s:e]

                curr_lr = session.run(opt.lr)
                grads[i] = opt.compute_gradient(x_batch, y_batch)
                # print(curr_lr, opt.compute_loss(x_batch, y_batch))

            opt.apply_gradients(*grads)

            # test
            if (epoch*num_minibatches + j) % check_loss_every == 0:
                if num_checks < 3:
                    num_checks += 1
                else:
                    return
                loss = opt.compute_loss(val[0], val[1])
                print(epoch, j, loss)
                if prev_loss is None:
                    prev_loss = loss
                else:
                    assert loss < prev_loss, "unexpected change in loss"


def test_simulate_distributed(seed=1337):
    # TODO: make this deterministic
    from tensorflow.examples.tutorials.mnist import input_data

    def get_data():
        # init data and minibatch parameters
        data = input_data.read_data_sets("MNIST_data", one_hot=True, seed=seed)
        train = (data.train.images, data.train.labels)
        val = data.validation.images, data.validation.labels
        test = data.test.images, data.test.labels
        return train, val, test

    train, val, test = get_data()
    N = len(train[0])
    n = worker_minibatch_size = 32
    k = num_workers = 2
    M = num_minibatches = int(math.ceil(N/(n*k)))

    # optimizer params
    init_eta = 0.1
    warmup_epochs = 5

    # init session
    session = tf.Session()

    # init server
    with tf.name_scope('server') as scope:
        server_model = LinearTFRayModel(session, shape=(784, 10))
        server_opt = TFGDRayOptimizer(session,
                                      server_model,
                                      num_minibatches,
                                      num_workers,
                                      worker_minibatch_size,
                                      init_learning_rate=init_eta,
                                      warmup_epochs=warmup_epochs)

    # init workers
    models = []
    opts = [None]*num_workers
    for i in range(num_workers):
        with tf.name_scope('worker_%d' % i) as scope:
            models.append(LinearTFRayModel(session, shape=(784, 10)))
            opts[i] = TFGDRayOptimizer(session,
                                       models[i],
                                       num_minibatches,
                                       num_workers,
                                       worker_minibatch_size,
                                       init_learning_rate=init_eta,
                                       warmup_epochs=warmup_epochs)

    # init variables
    session.run(tf.initialize_all_variables())

    server_model.weights = 1e-2 * np.random.RandomState(seed=seed).normal(size=server_model.num_weights)

    tf.summary.FileWriter('./tensorboard/multimodel', session.graph)

    # train
    total_epochs = 2
    shuffle_seed = seed
    prev_loss = None
    check_loss_every = 50
    num_checks = 0
    for epoch in range(total_epochs):
        shuffle_seed += 1
        idx = arg_shuffle(train[0], shuffle_seed)

        grads = [None]*num_workers
        for j in range(num_minibatches):

            # broadcast weights
            for model in models:
                model.weights = server_model.weights

            # compute gradients
            for i in range(num_workers):
                s, e = get_worker_batch(N, n, k, i, j)
                x_batch, y_batch = train[0][idx][s:e], train[1][idx][s:e]
                grads[i] = opts[i].compute_gradient(x_batch, y_batch)

            # aggregate gradients
            server_opt.apply_gradients(*grads)

            # test
            if (epoch*num_minibatches + j) % check_loss_every == 0:
                if num_checks < 3:
                    num_checks += 1
                else:
                    return
                loss = server_opt.compute_loss(val[0], val[1])
                print(epoch, j, session.run(server_opt.lr), loss)
                if prev_loss is None:
                    prev_loss = loss
                else:
                    assert loss < prev_loss, "unexpected change in loss"


if __name__ == "__main__":
    test_batching()
    test_opt_simple()
    test_linear_model()
    test_simulate_distributed()
