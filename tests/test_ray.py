import time
import pytest
import ray
import shared_memory_wrapper
from shared_memory_wrapper.util import interval_chunks

ray.init(num_cpus=1)
import numpy as np
RAY_DEDUP_LOGS = 0

@ray.remote
def func(array_ref, kmers_ref, start, end):
    t0 = time.perf_counter()
    result = array_ref[kmers_ref[start:end] % len(array_ref)]
    print("JOB time ", start, end, time.perf_counter() - t0)
    return result


def func2(array_ref, kmers_ref, start, end):
    t0 = time.perf_counter()
    result = array_ref[kmers_ref[start:end] % len(array_ref)]
    print("JOB time ", start, end, time.perf_counter() - t0)
    return result


#remote_func = ray.remote(func)


@pytest.mark.skip
def test():
    filter = np.zeros(2000000033, dtype=bool)
    kmers = np.random.randint(0, 2**63, 500000000, dtype=np.uint64)

    t0 = time.perf_counter()
    filter_ref = ray.put(filter)
    kmers_ref = ray.put(kmers)
    print(time.perf_counter()-t0)

    chunks = list(interval_chunks(0, len(kmers), 50))
    print(chunks)

    result = []
    for start, end in chunks:
        result.append(func.remote(filter_ref, kmers_ref, start, end))

    print(result)
    t0 = time.perf_counter()
    results = ray.get(result)
    print("Time parallel", time.perf_counter() - t0)
    print(results)

    t0 = time.perf_counter()
    result2 = func2(filter, kmers, 0, len(kmers))
    print("Timer nonparallel", time.perf_counter()-t0)

    t0 = time.perf_counter()
    result3 = func2(filter, kmers, 0, 10000000)
    print("Time 10 mill", time.perf_counter()-t0)

    #print(func(filter_ref, kmers_ref, 0, 100))




