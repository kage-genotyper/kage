import numpy as np
import pytest
from kage.indexing.kmer_scoring import MinCountBloomFilter


def test():
    modulos = [13, 9, 21, 41]
    keys = np.array([1, 2, 1, 4, 5, 4], dtype=np.uint64)

    counter = MinCountBloomFilter.empty(modulos)
    counter.count(keys)
    values = counter[keys]

    assert np.all(values == [2, 1, 2, 2, 1, 2])

    counter.count([1])
    assert np.all(counter[keys] == [3, 1, 3, 2, 1, 2])


