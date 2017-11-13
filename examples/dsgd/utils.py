#################################################
# Utility functions.
#################################################


import math
import random

import numpy as np


def arg_shuffle(arr, seed=None):
    if seed is not None:
        random.seed = seed
    r = np.arange(len(arr))
    random.shuffle(r)
    return r


class Batch(object):
    """
    Simple object for creating an object that can
    generate batches of sequential integers.
    """
    def __init__(self, total_size, batch_size):
        """
        :param total_size: Total number of items to split into batches.
        :param batch_size: Size of each batch.
        """
        self.total_size = total_size
        self.batch_size = batch_size
        self.batches = self.get_batches(total_size, batch_size)
        self.num_batches = len(self.batches)

    def get_batches(self, total_size, batch_size):
        """
        :param total_size: Total number of items to split into batches.
        :param batch_size: Size of each batch.
        :return: A list of 2-tuples.
                 Each 2-tuple is a segment of indices corresponding to items of size batch_size.
                 The size of the list is total_size / batch_size.
        """
        if total_size < batch_size:
            return [[0, total_size]]
        batches = list(range(0, total_size, batch_size))
        num_batches = int(total_size / batch_size)
        batches = [batches[i:i + 2] for i in range(0, num_batches, 1)]
        if len(batches[-1]) == 1:
            batches[-1].append(total_size)
        if batches[-1][1] != total_size:
            batches.append([batches[-1][1], total_size])
        return batches


# This is used to test get_worker_batch
def get_batches(N, n, k):
    worker_minibatch_size = n
    num_workers = k
    minibatch_size = num_workers * worker_minibatch_size
    batches = Batch(N, minibatch_size)

    last_minibatch_size = batches.batches[-1][1] - batches.batches[-1][0]

    worker_batches = Batch(minibatch_size, n)
    last_worker_batches = Batch(last_minibatch_size, n)

    last_batch_end = last_worker_batches.batches[-1][-1]
    for i in range(k):
        if i >= last_worker_batches.num_batches:
            last_worker_batches.batches.append([last_batch_end, last_batch_end])
    last_worker_batches.num_batches = len(last_worker_batches.batches)

    return batches, worker_batches, last_worker_batches


def get_worker_batch(N, n, k, i, j):
    """
    :param N: total number of training examples
    :param n: per-worker examples
    :param k: number of workers
    :param i: worker index
    :param j: minibatch index
    :return: A pair of integers indicating batch coordinates.
    """
    start_index = min(N, j*n*k + i*n)
    end_index = min(N, start_index + n)
    return start_index, end_index
    # if start_index == end_index:
    #     return None


if __name__ == "__main__":
    pass
