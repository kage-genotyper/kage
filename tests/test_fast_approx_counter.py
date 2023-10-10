import time

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


def test_add_many():
    counter = FastApproxCounter.empty(100)
    for i in range(1000):
        counter.add(np.array([1]))

    print(counter[1])


@pytest.mark.skip
def test_benchmark_parallel_add():
    counter = FastApproxCounter.empty(2000000003)
    kmers = np.random.randint(0, 2**63, 1000000, dtype=np.uint64)

    t0 = time.perf_counter()
    counter.add(kmers)
    print(time.perf_counter()-t0)

    counter2 = FastApproxCounter.empty(2000000003)
    t0 = time.perf_counter()
    counter2.add_numba(kmers)
    print(time.perf_counter()-t0)


    #counter3 = FastApproxCounter.empty(2000000003)
    #t0 = time.perf_counter()
    #counter3.add_numba2(kmers)
    #print(time.perf_counter()-t0)

    assert np.all(counter2.values == counter.values)

    #t0 = time.perf_counter()
    #counter.add_parallel(kmers, n_threads=8)
    #print(time.perf_counter()-t0)


if __name__ == "__main__":
    test_benchmark_parallel_add()
