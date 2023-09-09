import logging
logging.basicConfig(level=logging.INFO)
import time

import numpy as np
import numba
from kage.indexing.kmer_scoring import FastApproxCounter, MinCountBloomFilter



@numba.jit(nopython=True)
def numba_bincount(values, minlength):
    data = np.zeros(minlength, dtype=np.int32)
    for value in values:
        data[value] += 1

    return data


def test_benchmark_counter():
    modulo = 200000033 * 10
    n = 100000000 * 1
    kmers = np.random.randint(0, 2**63, n, dtype=np.uint64)
    """
    counter = FastApproxCounter.empty(modulo)
    t0 = time.perf_counter()
    counter.add(kmers)
    print("Time approx counter: %.5f" % (time.perf_counter() - t0))

    t0 = time.perf_counter()
    counts = counter[kmers]
    print("Time to get counts: %.5f" % (time.perf_counter() - t0))
    """


    # Min count bloom filter
    filter = MinCountBloomFilter.empty([10000001, 8000003, 12000033, 13000097])
    t0 = time.perf_counter()
    filter.count(kmers)
    print("Time bloom filter: %.5f" % (time.perf_counter() - t0))
    t0 = time.perf_counter()
    #counts = filter[kmers]
    print("Time to get counts: %.5f" % (time.perf_counter() - t0))



if __name__ == "__main__":
    test_benchmark_counter()