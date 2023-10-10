import time

import pytest
from kage.indexing.modulo_filter import ModuloFilter
import numpy as np


def test_modulo_filter():

    elements = np.array([1, 10, 15, 112839123])
    filter = ModuloFilter.empty(123)
    filter.add(elements)
    assert np.all(filter[elements])


@pytest.mark.skip
def test_benchmark_modulo_filter():
    modulo = 2_000_000_033
    n_kmers = 200000000
    filter = ModuloFilter(np.random.randint(0, 2, modulo, dtype=bool))
    kmers = np.random.randint(0, 2**63, n_kmers)
    filter.getitem_numba([1, 2])  # init numba

    t0 = time.perf_counter()
    result = filter[kmers]
    print(time.perf_counter() - t0)

    t0 = time.perf_counter()
    result2 = filter.getitem_numba(kmers)
    print(time.perf_counter() - t0)
    assert np.all(result == result2)
