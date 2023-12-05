import logging
logging.basicConfig(level=logging.DEBUG)
import time
import ray
import numpy as np
from kage.indexing.path_based_count_model import PathKmers, prune_kmers_parallel
from shared_memory_wrapper import from_file
import bionumpy as bnp

def main():
    n_threads = 1
    kmer_index = from_file("kmer_index_test.npz").copy()
    kmers = kmer_index.get_kmers()
    n_kmers = 100_000_000
    random_kmers = np.concatenate([kmers, np.random.randint(0, 2**63, n_kmers, dtype=np.uint64)])

    encoding = bnp.get_kmers(bnp.as_encoded_array("G"*31, bnp.DNAEncoding), 31).encoding
    n_rows = int(n_kmers / 10)

    kmers = bnp.EncodedRaggedArray(
        bnp.EncodedArray(random_kmers, encoding),
        np.zeros(n_rows, int) + 10
    )

    t0 = time.perf_counter()
    pruned = PathKmers.prune_kmers(kmers, kmer_index, n_threads=n_threads)
    print("Time", time.perf_counter()-t0)

    return
    # parallel
    ray.init(num_cpus=n_threads)
    t0 = time.perf_counter()
    pruned2 = prune_kmers_parallel(kmers, kmer_index, n_threads=n_threads)
    print("Time parallel", time.perf_counter()-t0)


if __name__ == "__main__":
    main()