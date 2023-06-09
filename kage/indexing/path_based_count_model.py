import logging
import time
from dataclasses import dataclass
import numpy as np
from scipy.signal import convolve2d
import bionumpy as bnp
from typing import List

from shared_memory_wrapper import to_file

from .path_variant_indexing import Graph, MappingModelCreator, Paths
from graph_kmer_index import KmerIndex
from .sparse_haplotype_matrix import SparseHaplotypeMatrix
from ..models.mapping_model import LimitedFrequencySamplingComboModel
import npstructures as nps

@dataclass
class HaplotypeAsPaths:
    """Represents a haplotype as a combination of paths"""
    paths: np.ndarray


    @classmethod
    def from_haplotype_and_path_alleles(cls, haplotype: np.ndarray, path_alleles: np.ndarray, window):
        convolve_array = 2**np.arange(window)
        # get a signature for every window at every path
        # will match the haplotype to paths by matching the signatures at each variant

        # todo: these signatures shouldn't be created every time but can be premade
        path_signatures = convolve2d(path_alleles, [convolve_array], mode='full')[:,window-1:].astype(np.uint16)

        haplotype_signatures = np.convolve(haplotype, convolve_array, mode='full')[window-1:].astype(np.uint16)

        matching_paths = np.zeros(len(haplotype), dtype=np.uint16)
        for i in range(path_signatures.shape[0]):
            matching_paths[path_signatures[i] == haplotype_signatures] = i
        return cls(matching_paths)

    @classmethod
    def _from_haplotype_and_path_alleles(cls, haplotype: np.ndarray, path_alleles: np.ndarray, window):
        # for every variant, find the path that matches the haplotype in the window from that variant
        # paths are repeting, so the first window variant can be used
        path_signatures = path_alleles[:, :window]
        convolve_array = 2**np.arange(window)
        path_signatures = convolve2d(path_signatures, [convolve_array], mode='valid').ravel()
        signature_to_path = np.zeros(path_alleles.shape[0])
        signature_to_path[path_signatures] = np.arange(path_signatures.shape[0], dtype=np.uint8)
        haplotype_window_signatures = np.convolve(haplotype, convolve_array, mode='full')[window-1:].astype(np.uint8)
        return cls(signature_to_path[haplotype_window_signatures])



@dataclass
class PathKmers:
    kmers: List[bnp.EncodedRaggedArray]

    @classmethod
    def from_graph_and_paths(cls, graph: Graph, path_allele_matrix: np.ndarray, k):
        # kmers for a variant should include the kmers for the variant and
        # the kmers for the following ref sequence before next variant
        n_paths = path_allele_matrix.shape[0]
        logging.info("Making pathkmers for %d paths", n_paths)
        return cls((graph.kmers_for_pairs_of_ref_and_variants(path_allele_matrix[i, :], k) for i in range(n_paths)))


    def get_for_haplotype(self, haplotype: HaplotypeAsPaths, include_reverse_complements=False):
        """
        Returns all kmers for this haplotype (by using the fact that the haplotype is
        a combination of paths).

        Returns kmers in no particular order.
        """
        if not isinstance(self.kmers, list):
            # only used for testing. This function is normally called after pruning and then kmers is a list
            self.kmers = list(self.kmers)

        kmers_found = []
        # add kmers on first ref node
        start_kmers = self.kmers[int(haplotype.paths[0])][0].ravel()
        encoding = start_kmers.encoding
        kmers_found.append(start_kmers.raw())

        for path in range(len(self.kmers)):
            print("PATH ", path)
            print(haplotype, path, len(self.kmers[path]))
            variants_with_path = np.where(haplotype.paths == path)[0] + 1  # +1 because first element in kmers is the first ref

            print("Variants with path")
            print(variants_with_path)
            print(len(self.kmers[path]))
            print(np.sum(self.kmers[path].shape[1]))
            print("All path kmers")
            print(len(self.kmers[path].ravel()))

            print("Subset")
            kmers = self.kmers[path][variants_with_path]
            print(kmers.ravel())
            print(len(kmers))
            print(kmers.shape)
            print(kmers.dtype)
            kmers = kmers.raw().ravel()
            kmers_found.append(kmers)

        return bnp.EncodedArray(np.concatenate(kmers_found), encoding)


    def prune(self, kmer_index):
        """
        Prunes away kmers that are not in kmer index.
        Prunes inplace.
        """
        new = []
        for i, kmers in enumerate(self.kmers):
            assert np.all(kmers.shape[0] >= 0)
            print("----------__I ", i)
            encoding = kmers.encoding
            raw_kmers = kmers.raw().ravel().astype(np.uint64)
            is_in = kmer_index.has_kmers(raw_kmers)
            print("IS IN")
            print(is_in)
            assert len(is_in) == len(kmers.ravel()) == len(raw_kmers) == np.sum(kmers.shape[1])
            mask = nps.RaggedArray(is_in, kmers.shape, dtype=bool)
            print("SUM IS IN: %d" % np.sum(is_in))
            print("SUM mask: %d" % np.sum(mask))
            print("SUM mask ravel: %d" % np.sum(mask.ravel()))
            print("SUM mask row sums: %d" % np.sum(np.sum(mask, axis=1)))
            print("N raw kmers: %d" % len(raw_kmers))
            print("N raw kmers[mask]: %d" % len(raw_kmers[mask.ravel()]))

            #if i == 2:
            #    to_file(mask, "mask.npy")
            #    assert False

            assert len(mask.ravel()) == np.sum(mask.shape[1])
            assert len(mask.ravel()) == len(is_in)
            assert np.sum(mask.ravel()) == np.sum(is_in)
            logging.info(f"Pruned away {np.sum(mask==False)}/{len(kmers)} kmers for path {i}")
            kmers = raw_kmers[mask.ravel()]
            shape = np.sum(mask, axis=1)
            print("Shape for path ", i, ":", np.sum(shape))
            assert np.sum(shape) == len(kmers), (np.sum(shape), len(kmers), np.sum(is_in), np.sum(is_in == True))
            new.append(bnp.EncodedRaggedArray(bnp.EncodedArray(kmers, encoding), shape))
            assert np.sum(mask) == len(kmers)
        self.kmers = new

class PathBasedMappingModelCreator(MappingModelCreator):
    def __init__(self, graph: Graph, kmer_index: KmerIndex,
                 haplotype_matrix: SparseHaplotypeMatrix, window, paths_allele_matrix = None,
                 max_count=10, k=31):
        self._graph = graph
        self._kmer_index = kmer_index
        self._haplotype_matrix = haplotype_matrix
        self._n_nodes = graph.n_nodes()
        self._counts = LimitedFrequencySamplingComboModel.create_empty(self._n_nodes, max_count)
        logging.info("Inited empty model")
        self._k = k
        self._max_count = max_count
        self._path_allele_matrix = paths_allele_matrix
        self._path_kmers = PathKmers.from_graph_and_paths(graph, paths_allele_matrix, k=k)
        self._path_kmers.prune(kmer_index)

    def _process_individual(self, i):
        t0 = time.perf_counter()
        #logging.info("Staring individual %d", i)
        haplotype1 = self._haplotype_matrix.get_haplotype(i * 2)
        haplotype2 = self._haplotype_matrix.get_haplotype(i * 2 + 1)

        #logging.info("Getting haplotypes took %.3f sec", time.perf_counter() - t0)
        t0 = time.perf_counter()

        all_kmers = []
        for haplotype in [haplotype1, haplotype2]:
            as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles(haplotype, self._path_allele_matrix, window=3)
            all_kmers.append(self._path_kmers.get_for_haplotype(as_paths).raw().ravel().astype(np.uint64))

        #logging.info("Getting all kmers took %.3f sec", time.perf_counter() - t0)
        t0 = time.perf_counter()

        haplotype1_nodes = self._haplotype_matrix.get_haplotype_nodes(i*2)
        haplotype2_nodes = self._haplotype_matrix.get_haplotype_nodes(i*2+1)
        #logging.info("Getting nodes took %.3f sec", time.perf_counter() - t0)
        t0 = time.perf_counter()

        node_counts = self._kmer_index.map_kmers(np.concatenate(all_kmers), self._n_nodes)
        #logging.info("Mapping kmers took %.3f sec", time.perf_counter() - t0)
        t0 = time.perf_counter()

        self._add_node_counts(haplotype1_nodes, haplotype2_nodes, node_counts)
        #logging.info("Adding node counts took %.3f sec", time.perf_counter() - t0)


