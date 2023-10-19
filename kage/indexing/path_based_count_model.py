import logging
import time
from dataclasses import dataclass
import numpy as np
import ray
from shared_memory_wrapper.util import interval_chunks

from kage.preprocessing.variants import VariantAlleleToNodeMap
from scipy.signal import convolve2d
import bionumpy as bnp
from typing import List, Union

from shared_memory_wrapper import to_file, object_to_shared_memory, object_from_shared_memory

from .modulo_filter import ModuloFilter
from .path_variant_indexing import MappingModelCreator
from kage.indexing.graph import Graph
from .paths import Paths
from graph_kmer_index import KmerIndex
from .sparse_haplotype_matrix import SparseHaplotypeMatrix
from ..models.mapping_model import LimitedFrequencySamplingComboModel
import npstructures as nps
from ..util import log_memory_usage_now
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

@dataclass
class HaplotypeAsPaths:
    """Represents a haplotype as a combination of paths"""
    paths: np.ndarray


    @classmethod
    def from_haplotype_and_path_alleles_multiallelic(cls, haplotype: np.ndarray, path_alleles: np.ndarray, window):
        # same as method below, but supports haplotype and path alleles that are multiallelic
        # thus, cannot match using convolve. This should support biallelic nicely also
        # idea is to use sliding window and pad first with 0s, and do matching of windows

        # padd with 0s at end so that last window can be matched
        path_signatures = sliding_window_view(
            np.append(path_alleles, np.zeros((path_alleles.shape[0], window-1)), axis=1),
            window, axis=1)
        haplotype_signatures = sliding_window_view(np.append(haplotype, np.zeros(window-1)), window)
        matching_paths = np.zeros(len(haplotype), dtype=np.uint16)

        #haplotype_signatures = np.append(haplotype_signatures, np.zeros(window-1))
        #path_signatures = np.append(path_signatures, np.zeros((path_signatures.shape[0], window-1)), axis=1)

        for i in range(path_signatures.shape[0]):
            matching_paths[np.all(path_signatures[i] == haplotype_signatures, axis=1)] = i
        return cls(matching_paths)

    @classmethod
    def from_haplotype_and_path_alleles(cls, haplotype: np.ndarray, path_alleles: np.ndarray, window):
        return cls.from_haplotype_and_path_alleles_multiallelic(haplotype, path_alleles, window)
        # todo: can be removed
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


@dataclass
class PathKmers:
    kmers: List[bnp.EncodedRaggedArray]

    @classmethod
    def from_graph_and_paths(cls, graph: Graph, path_allele_matrix: np.ndarray, k):
        # kmers for a variant should include the kmers for the variant and
        # the kmers for the following ref sequence before next variant
        n_paths = path_allele_matrix.shape[0]
        log_memory_usage_now("Before making pathkmers")
        logging.info("Making pathkmers for %d paths", n_paths)
        return cls((graph.kmers_for_pairs_of_ref_and_variants(path_allele_matrix[i, :], k) for i in range(n_paths)))

    def get_for_haplotype(self, haplotype: HaplotypeAsPaths, include_reverse_complements=False):
        """
        Returns all kmers for this haplotype (by using the fact that the haplotype is
        a combination of paths).

        Returns kmers in no particular order.
        """
        if not isinstance(self.kmers, list):
            self.kmers = list(self.kmers)

        kmers_found = []
        # add kmers on first ref node
        start_kmers = self.kmers[int(haplotype.paths[0])][0].ravel()
        encoding = start_kmers.encoding
        kmers_found.append(start_kmers.raw())

        for path in range(len(self.kmers)):
            variants_with_path = np.where(haplotype.paths == path)[0] + 1  # +1 because first element in kmers is the first ref
            kmers = self.kmers[path][variants_with_path]
            kmers = kmers.raw().ravel()
            kmers_found.append(kmers)

        return bnp.EncodedArray(np.concatenate(kmers_found), encoding)


    def prune(self, kmer_index, n_threads=1, modulo=2_000_000_033):
        """
        Prunes away kmers that are not in kmer index.
        Prunes inplace.
        Modulo used when creating approx lookup
        """
        # first creates an approx lookup of which kmers are in kmer_index
        # then prunes away kmers that are not in kmer_index
        logging.info("Making kmer lookup")
        lookup = kmer_index
        #index_kmers = kmer_index.get_kmers()
        #lookup = np.zeros(modulo, dtype=bool)
        #lookup[index_kmers % modulo] = True
        #lookup = ModuloFilter(lookup)

        #logging.info("Pruning")
        new = []
        for i, kmers in tqdm(enumerate(self.kmers), desc="Pruning kmers"):
            #logging.info("Pruning path %d", i)
            t0 = time.perf_counter()
            pruned_kmers = self.prune_kmers(kmers, lookup)
            new.append(pruned_kmers)
            #logging.info("Pruning took %.5f sec" % (time.perf_counter() - t0))

        self.kmers = new

    @staticmethod
    def prune_kmers(kmers: bnp.EncodedRaggedArray, lookup: Union[ModuloFilter, KmerIndex], n_threads=1):
        assert np.all(kmers.shape[0] >= 0)
        encoding = kmers.encoding
        raw_kmers = kmers.raw().ravel().astype(np.uint64)
        #logging.info("Got %d raw kmers" % len(raw_kmers))
        t_lookup = time.perf_counter()
        if isinstance(lookup, ModuloFilter):
            is_in = lookup[raw_kmers]
        else:
            if n_threads == 1:
                is_in = lookup.has_kmers(raw_kmers)
            else:
                logging.warning("Experimentatl with more than 1 thread. Probably not faster.")
                is_in = parallel_kmer_index_has_kmers(raw_kmers, lookup)

        #logging.info("Time lookup %d kmers: %.5f" % (len(raw_kmers), time.perf_counter() - t_lookup))
        mask = nps.RaggedArray(is_in, kmers.shape, dtype=bool)
        #print(f"Kept {np.sum(mask)}/{len(raw_kmers)} kmers for path")
        kmers = raw_kmers[is_in]
        shape = np.sum(mask, axis=1)
        pruned_kmers = bnp.EncodedRaggedArray(bnp.EncodedArray(kmers, encoding), shape)
        return pruned_kmers



def _kmer_index_has_kmers_chunk(kmer_index, kmers, out_array, interval):
    start, end = interval
    t0 = time.perf_counter()
    out_array[start:end] = kmer_index.has_kmers(kmers[start:end])
    print("Work took ", time.perf_counter()-t0)


def parallel_kmer_index_has_kmers(kmers, kmer_index, n_threads=8):
    """
    Experimental: Does not seem to be any faster than nonparallel, too much overhead
    and probably too many threads accessing same memory
    """
    from shared_memory_wrapper.util import parallel_map_reduce, interval_chunks
    t0 = time.perf_counter()
    kmer_index = kmer_index.copy()
    out_array = np.zeros_like(kmers, dtype=bool)
    chunks = interval_chunks(0, len(kmers), n_threads)
    print("Time to init", time.perf_counter()-t0)

    data = parallel_map_reduce(_kmer_index_has_kmers_chunk, (kmer_index, kmers, out_array),
                        chunks,
                        n_threads=n_threads)
    out_array = data[2]
    return out_array
    #return kmer_index.has_kmers(kmers)


@ray.remote
def _prune(kmers, lookup, start, end):
    t0 = time.perf_counter()
    result = PathKmers.prune_kmers(kmers[start:end], lookup)
    print("Time for ", start, end, ":", time.perf_counter()-t0)
    return result


def prune_kmers_parallel(kmers, lookup, n_threads=8):
    #ray.init(num_cpus=n_threads)
    chunks = interval_chunks(0, len(kmers), n_threads)
    print(chunks)
    t0 = time.perf_counter()
    kmers = ray.put(kmers)
    lookup = ray.put(lookup)
    print("Putting data took ", time.perf_counter()-t0)

    result = []
    for start, end in chunks:
        result.append(_prune.remote(kmers, lookup, start, end))

    result = ray.get(result)
    t0 = time.perf_counter()
    result = np.concatenate(result)
    print("Time to concatenate: ", time.perf_counter()-t0)
    return result


class PathBasedMappingModelCreator(MappingModelCreator):
    def __init__(self, graph: Graph, kmer_index: KmerIndex,
                 haplotype_matrix: SparseHaplotypeMatrix, window, paths_allele_matrix = None,
                 max_count=10, k=31, node_map: VariantAlleleToNodeMap = None, n_nodes: int = None,
                 n_threads=16):
        self._graph = graph
        self._kmer_index = kmer_index
        self._haplotype_matrix = haplotype_matrix
        self._n_nodes = n_nodes
        if n_nodes is None:
            logging.warning("N nodes not specified. Getting from graph")
            self._n_nodes = graph.n_nodes()

        self._counts = LimitedFrequencySamplingComboModel.create_empty(self._n_nodes, max_count)
        logging.info("Inited empty model")
        self._k = k
        self._max_count = max_count
        self._path_allele_matrix = paths_allele_matrix
        self._path_kmers = PathKmers.from_graph_and_paths(graph, paths_allele_matrix, k=k)
        self._path_kmers.prune(kmer_index, n_threads=16)
        self._node_map = node_map
        self._window = window
        self._all_haplotypes_as_paths = get_haplotypes_as_paths_parallel(
            self._haplotype_matrix, self._path_allele_matrix, self._window, n_threads=n_threads)

    def _process_individual(self, i):
        t0 = time.perf_counter()
        all_haplotypes_as_paths = self._all_haplotypes_as_paths

        haplotype1 = self._haplotype_matrix.get_haplotype(i * 2)
        haplotype2 = self._haplotype_matrix.get_haplotype(i * 2 + 1)
        #logging.info("Getting haplotypes took %.5f sec" % (time.perf_counter() - t0))

        all_kmers = []
        #for haplotype_id, haplotype in enumerate([haplotype1, haplotype2]):
        for haplotype_id in [0, 1]:
            t0 = time.perf_counter()
            #as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles_multiallelic(haplotype, self._path_allele_matrix,
            #                                                                         window=self._window)
            as_paths = all_haplotypes_as_paths[i*2 + haplotype_id]
            #print(len(haplotype), self._path_allele_matrix.shape)
            #logging.info("As paths took %.5f" % (time.perf_counter()-t0))
            #print("Individual %d, haplotype %s as paths: %s" % (i, haplotype, as_paths))
            t0 = time.perf_counter()
            all_kmers.append(self._path_kmers.get_for_haplotype(as_paths).raw().ravel().astype(np.uint64))
            #logging.info("Getting kmers took %.5f sec" % (time.perf_counter() - t0))

        # todo for multiallelic: use variant_to_nodes
        t0 = time.perf_counter()
        if self._node_map is None:
            logging.info("No node map provided. Assuming biallelic and deciding node ids implicitly")
            haplotype1_nodes = self._haplotype_matrix.get_haplotype_nodes(i*2)
            haplotype2_nodes = self._haplotype_matrix.get_haplotype_nodes(i*2+1)
        else:
            haplotype1_nodes = self._node_map.haplotypes_to_node_ids(haplotype1)
            haplotype2_nodes = self._node_map.haplotypes_to_node_ids(haplotype2)
        #logging.info("Getting node ids took %.5f sec" % (time.perf_counter() - t0))

        t0 = time.perf_counter()
        node_counts = self._kmer_index.map_kmers(np.concatenate(all_kmers), self._n_nodes)
        #logging.info("Mapping took %.5f sec" % (time.perf_counter() - t0))
        t0 = time.perf_counter()
        self._add_node_counts(haplotype1_nodes, haplotype2_nodes, node_counts)
        #logging.info("Adding counts took %.5f sec" % (time.perf_counter() - t0))


def get_haplotypes_as_paths(haplotype_matrix: SparseHaplotypeMatrix, path_allele_matrix: np.ndarray, window_size):
    """
    Returns all haplotypes as paths by matching against path_allele_matrix
    """
    t0 = time.perf_counter()
    out = []
    path_signatures = sliding_window_view(
        np.append(path_allele_matrix, np.zeros((path_allele_matrix.shape[0], window_size - 1)), axis=1),
        window_size, axis=1)

    for i in range(haplotype_matrix.n_haplotypes):
        haplotype = haplotype_matrix.get_haplotype(i)

        haplotype_signatures = sliding_window_view(np.append(haplotype, np.zeros(window_size-1)), window_size)
        matching_paths = np.zeros(len(haplotype), dtype=np.uint16)

        for i in range(path_signatures.shape[0]):
            matching_paths[np.all(path_signatures[i] == haplotype_signatures, axis=1)] = i

        out.append(HaplotypeAsPaths(matching_paths))

    logging.info("Getting all haplotypes as paths took %.5f sec" % (time.perf_counter() - t0))
    return out

@ray.remote
def _get_single_haplotype_as_paths(path_signatures, haplotype_matrix, haplotype_id, window_size):
    log_memory_usage_now("_get_single_haplotype_as_paths_start")
    haplotype = haplotype_matrix.get_haplotype(haplotype_id)

    haplotype_signatures = sliding_window_view(np.append(haplotype, np.zeros(window_size - 1)), window_size)
    matching_paths = np.zeros(len(haplotype), dtype=np.uint8)
    assert path_signatures.shape[0] <= 256, "Too many paths for uint8"

    for i in range(path_signatures.shape[0]):
        matching_paths[np.all(path_signatures[i] == haplotype_signatures, axis=1)] = i
    log_memory_usage_now("_get_single_haplotype_as_paths_end")

    return HaplotypeAsPaths(matching_paths)


def get_haplotypes_as_paths_parallel(haplotype_matrix: SparseHaplotypeMatrix, path_allele_matrix: np.ndarray,
                                     window_size, n_threads=8):
    t0 = time.perf_counter()
    ray.init(num_cpus=n_threads, ignore_reinit_error=True)
    print("Time init", time.perf_counter()-t0)

    out = []
    t0 = time.perf_counter()
    n_haplotypes = haplotype_matrix.n_haplotypes
    haplotype_matrix = ray.put(haplotype_matrix)

    path_signatures = sliding_window_view(
        np.append(path_allele_matrix, np.zeros((path_allele_matrix.shape[0], window_size - 1)), axis=1),
        window_size, axis=1)

    path_signatures = ray.put(path_signatures)
    print("Time putting things in memory etc", time.perf_counter()-t0)

    for haplotype in range(n_haplotypes):
        out.append(_get_single_haplotype_as_paths.remote(path_signatures, haplotype_matrix, haplotype, window_size))

    t0 = time.perf_counter()
    results = ray.get(out)
    print("Time doing stuff", time.perf_counter()-t0)

    return results