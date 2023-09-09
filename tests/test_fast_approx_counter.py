import numpy as np
import pytest
from kage.indexing.kmer_scoring import FastApproxCounter


def test_fast_approx_counter_add():
    counter = FastApproxCounter.empty(100)
    counter.add(np.array([1, 1, 16, 4, 3]))

    assert counter.values[1] == 2
    assert counter.values[16] == 1


def test_fast_approx_counter_add_parallel():
    counter = FastApproxCounter.empty(100)
    counter.add_parallel(np.array([1, 1, 16, 4, 3]), n_threads=3)
    print(counter.values)

    assert counter.values[1] == 2
    assert counter.values[16] == 1
    assert counter.values[4] == 1
    assert counter.values[3] == 1
