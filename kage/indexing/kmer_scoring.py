import itertools
import logging
import time
from typing import List

import npstructures as nps
import numba
import numpy as np
import shared_memory_wrapper.util
from numba import njit, prange
from shared_memory_wrapper.util import interval_chunks
from shared_memory_wrapper.shared_array_map_reduce import additative_shared_array_map_reduce
import tqdm
from kage.indexing.graph import Graph
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.util import log_memory_usage_now


class FastApproxCounter:
    """ Fast counter that uses modulo and allows collisions"""
    def __init__(self, array, modulo):
        self._array = array
        self._modulo = modulo

    def copy(self):
        return FastApproxCounter(self._array.copy(), self._modulo)

    @classmethod
    def empty(cls, modulo, dtype=np.uint8):
        assert dtype == np.uint8, "Only uint8 support now"
        return cls(np.zeros(modulo, dtype=dtype), modulo)

    def add(self, values):
        value_hashes = (values % self._modulo).astype(int)
        t0 = time.perf_counter()
        add_counts = np.bincount(value_hashes, minlength=self._modulo)
        #print("Addding %d kmers with modulo %d took %.5f sec" % (len(values), self._modulo, time.perf_counter() - t0))
        self._array += np.minimum(255 - self._array, add_counts).astype(self._array.dtype)  # to not overflow

    def add_numba(self, values):
        value_hashes = (values % self._modulo).astype(int)
        t0 = time.perf_counter()

        @numba.jit(nopython=True)
        def _add(array, value_hashes):
            for value_hash in value_hashes:
                if array[value_hash] < 255:
                    array[value_hash] += 1

        _add(self._array, value_hashes)
        #print("Addding %d kmers with modulo %d took %.5f sec" % (len(values), self._modulo, time.perf_counter() - t0))

    def add_numba2(self, values):
        value_hashes = (values % self._modulo).astype(int)
        t0 = time.perf_counter()

        @njit
        def _add(array, value_hashes):
            for i in prange(len(value_hashes)): #value_hash in value_hashes:
                if array[value_hashes[i]] < 255:
                    array[value_hashes[i]] += 1

        _add(self._array, value_hashes)
        print("Addding %d kmers with modulo %d took %.5f sec" % (len(values), self._modulo, time.perf_counter() - t0))

    def add_parallel(self, values, n_threads=16):
        chunks = interval_chunks(0, len(values), len(values)//1000000+1)
        result_array = np.zeros(self._modulo, dtype=np.uint8)
        data = (values, self._modulo)

        def func(values, modulo, chunk):
            # will process a chunk of the values
            t0 = time.perf_counter()
            values = values[chunk[0]:chunk[1]]
            value_hashes = (values % modulo).astype(int)
            add_counts = np.bincount(value_hashes, minlength=modulo)
            print("  Counting %d values for chunk %d-%d, took %.5f sec" % (len(values), chunk[0], chunk[1], time.perf_counter()-t0))
            return add_counts

        def add_func(result_array, job_result):
            result_array += np.minimum(255 - result_array, job_result).astype(np.uint8)

        counts = additative_shared_array_map_reduce(func, chunks, result_array, data, n_threads=n_threads, queue_size_factor=0.5,
                                                    add_func=add_func)
        self._array += counts.astype(np.uint8)

    @property
    def values(self):
        return self._array

    @classmethod
    def from_keys_and_values(cls, keys, values, modulo):
        array = np.zeros(modulo, dtype=np.int32)
        array[keys % modulo] = values
        return cls(array, modulo)

    def __getitem__(self, keys):
        return self._array[keys % self._modulo]

    def score_kmers(self, kmers):
        return -self[kmers].astype(int)


def make_kmer_scorer_from_random_haplotypes(graph: Graph, haplotype_matrix: SparseHaplotypeMatrix,
                                            k: int,
                                            n_haplotypes: int = 4,
                                            modulo: int = 20000033):
    """
    Estimates counts from random individuals
    """
    log_memory_usage_now("Memory before making kmer scorer")
    counter = FastApproxCounter.empty(modulo)

    if n_haplotypes > haplotype_matrix.n_haplotypes:
        logging.info("Limiting to %d haplotypes, since that is what is in population" % haplotype_matrix.n_haplotypes)
        n_haplotypes = haplotype_matrix.n_haplotypes

    chosen_haplotypes = np.random.choice(np.arange(haplotype_matrix.n_haplotypes), n_haplotypes, replace=False)
    logging.info("Picked random haplotypes to make kmer scorer: %s" % chosen_haplotypes)
    haplotype_nodes = (haplotype_matrix.get_haplotype(haplotype) for haplotype in chosen_haplotypes)

    # also add the reference and a haplotype with all variants
    haplotype_nodes = itertools.chain(haplotype_nodes,
                                      [np.zeros(haplotype_matrix.n_variants, dtype=np.uint8),
                                       np.ones(haplotype_matrix.n_variants, dtype=np.uint8)
                                       ])

    for i, nodes in tqdm.tqdm(enumerate(haplotype_nodes), desc="Estimating global kmer counts", total=len(chosen_haplotypes), unit="haplotype"):
        #log_memory_usage_now("Memory after getting nodes")

        for reverse_complement in [False, True]:
            #logging.info("Reverse complement %s" % reverse_complement)
            t0 = time.perf_counter()
            kmers = graph.get_haplotype_kmers(nodes, k=k, stream=True, reverse_complement=reverse_complement)
            #logging.info("Getting kmers took %.5f sec" % (time.perf_counter() - t0))
            #log_memory_usage_now("Memory after kmers")
            t0 = time.perf_counter()
            t_add = 0
            n_kmers = 0
            for subkmers in kmers:
                t_add_start = time.perf_counter()
                counter.add_numba(subkmers)
                t_add += time.perf_counter() - t_add_start
                n_kmers += len(subkmers)
            #logging.info("Adding %d kmers to counter took %.5f sec" % (n_kmers, t_add))
            #logging.info("Tot time: %.5f" % (time.perf_counter() - t0))

        #log_memory_usage_now("After adding haplotype %d" % i)

    return counter


class Scorer:
    def __init__(self):
        pass

    def score_kmers(self, kmers):
        pass


class KmerFrequencyScorer(Scorer):
    def __init__(self, frequency_lookup: nps.HashTable):
        self._frequency_lookup = frequency_lookup

    def score_kmers(self, kmers):
        return np.max(self._frequency_lookup[kmers])


class MinCountBloomFilter:
    def __init__(self, data: np.ndarray, modulos: List[int]):
        # data is a RaggedArray where each row corresponds to a modulo
        self._flat_data = data
        self._modulos = np.array([[m] for m in modulos], dtype=np.uint32)
        self._starts = np.array([[v] for v in np.cumsum([0] + modulos)[:-1]], dtype=np.uint32)

    def hashes(self, keys):
        # modulo with colum creates matrix, add cumsum of modulos to get index in flat RaggedArray
        hashes = ((keys % self._modulos) + self._starts).ravel()
        return hashes

    def __getitem__(self, keys: np.ndarray) -> np.ndarray:
        hashes = self.hashes(keys)
        values = self._flat_data[hashes].reshape(-1, len(keys))
        return np.min(values, axis=0)

    def count(self, keys):
        hashes = self.hashes(keys).astype(np.int64)
        to_add = np.bincount(hashes, minlength=len(self._flat_data)).astype(np.uint32)
        self._flat_data += to_add

    @classmethod
    def empty(cls, modulos):
        size = sum(modulos)
        print("Total size: %d" % size)
        return cls(np.zeros(size, dtype=np.uint32), modulos)

