import itertools
import logging

import npstructures as nps
import numpy as np
import tqdm
from kage.indexing.graph import Graph
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.util import log_memory_usage_now
from graph_kmer_index.kmer_hashing import kmer_hashes_to_reverse_complement_hash

class FastApproxCounter:
    """ Fast counter that uses modulo and allows collisions"""
    def __init__(self, array, modulo):
        self._array = array
        self._modulo = modulo

    @classmethod
    def empty(cls, modulo):
        return cls(np.zeros(modulo, dtype=np.int16), modulo)

    def add(self, values):
        value_hashes = values % self._modulo
        add_counts = np.bincount(value_hashes.astype(int), minlength=self._modulo).astype(np.int16)
        self._array += add_counts

    @classmethod
    def from_keys_and_values(cls, keys, values, modulo):
        array = np.zeros(modulo, dtype=np.int16)
        array[keys % modulo] = values
        return cls(array, modulo)

    def __getitem__(self, keys):
        return self._array[keys % self._modulo]

    def score_kmers(self, kmers):
        return -self[kmers]


def make_kmer_scorer_from_random_haplotypes(graph: Graph, haplotype_matrix: SparseHaplotypeMatrix,
                                            k: int,
                                            n_haplotypes: int = 4,
                                            modulo: int = 20000033):
    """
    Estimates counts from random individuals
    """
    counter = FastApproxCounter.empty(modulo)
    chosen_haplotypes = np.random.choice(np.arange(haplotype_matrix.n_haplotypes), n_haplotypes, replace=False)
    logging.info("Picked random haplotypes to make kmer scorer: %s" % chosen_haplotypes)
    print("Chosen haplotypes", chosen_haplotypes)
    haplotype_nodes = (haplotype_matrix.get_haplotype(haplotype) for haplotype in chosen_haplotypes)

    # also add the reference and a haplotype with all variants
    haplotype_nodes = itertools.chain(haplotype_nodes,
                                      [np.zeros(haplotype_matrix.n_variants, dtype=np.uint8),
                                        np.ones(haplotype_matrix.n_variants, dtype=np.uint8)])

    for i, nodes in tqdm.tqdm(enumerate(haplotype_nodes), desc="Estimating global kmer counts", total=len(chosen_haplotypes), unit="haplotype"):
        #haplotype_nodes = haplotype_matrix.get_haplotype(haplotype)
        log_memory_usage_now("Memory after getting nodes")
        kmers = graph.get_haplotype_kmers(nodes, k=k, stream=True)
        log_memory_usage_now("Memory after kmers")
        for subkmers in kmers:
            counter.add(subkmers)
            # also add reverse complement
            subkmers_revcomp = kmer_hashes_to_reverse_complement_hash(subkmers, k)
            counter.add(subkmers_revcomp)

        log_memory_usage_now("After adding haplotype %d" % i)

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
