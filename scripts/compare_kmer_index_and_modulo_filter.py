import time
import ray
import shared_memory_wrapper
from graph_kmer_index import KmerIndex
from shared_memory_wrapper import from_file, object_to_shared_memory, object_from_shared_memory
import numpy as np
from shared_memory_wrapper.util import interval_chunks

from kage.indexing.modulo_filter import ModuloFilter


def main():
    kmer_index = from_file("kmer_index_test.npz")
    kmers = kmer_index.get_kmers()
    modulo_filter = ModuloFilter.empty(2_000_000_003)
    modulo_filter.add(kmers)

    n_kmers = 200_000_000
    #random_kmers = np.concatenate([kmers, np.random.randint(0, 2**63, n_kmers, dtype=np.uint64)])
    random_kmers = np.random.randint(0, 2**63, n_kmers, dtype=np.uint64)
    print("Made random kmers")

    """
    t0 = time.perf_counter()
    match = modulo_filter[random_kmers]
    print("Modulo filter", time.perf_counter()-t0)
    print(np.sum(match))
    """

    t0 = time.perf_counter()
    match2 = kmer_index.has_kmers(random_kmers)
    print("Kmer index", time.perf_counter()-t0)
    print(np.sum(match2))

    """
    ray.init(num_cpus=4)
    kmer_index = object_to_shared_memory(kmer_index)
    t0 = time.perf_counter()
    match3 = kmer_index_has_kmers_parallel(kmer_index, random_kmers, 8)
    print("Kmer index parallel", time.perf_counter()-t0)
    print(np.sum(match3))
    """

@ray.remote
def kmer_index_has_kmers(kmer_index_name, kmers, start, end):
    kmer_index = object_from_shared_memory(kmer_index_name)
    kmers = object_from_shared_memory(kmers)
    #subkmers = kmers[start:end].copy()
    t0 = time.perf_counter()
    res = kmer_index.has_kmers(kmers[start:end])
    print("Chunk of %d kmers took %.5f" % (len(kmers[start:end]), time.perf_counter()-t0))
    return res


def kmer_index_has_kmers_parallel(kmer_index_shm, kmers, n_threads=8):
    n_chunks = n_threads
    chunks = interval_chunks(0, len(kmers), n_chunks)

    t0 = time.perf_counter()
    kmers = object_to_shared_memory(kmers)
    print("Putting in shared memory", time.perf_counter()-t0)

    results = []
    for start, end in chunks:
        results.append(kmer_index_has_kmers.remote(kmer_index_shm, kmers, start, end))

    result = ray.get(results)
    t0 = time.perf_counter()
    result = np.concatenate(result)
    print("Concatenating result: ", time.perf_counter()-t0)
    shared_memory_wrapper.free_memory_in_session()
    return result



if __name__ == "__main__":
    main()
