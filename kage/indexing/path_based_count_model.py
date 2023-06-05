from dataclasses import dataclass
import numpy as np
from scipy.signal import convolve2d
import bionumpy as bnp
from typing import List
from .path_variant_indexing import Graph


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
        return cls([graph.kmers_for_pairs_of_ref_and_variants(path_allele_matrix[i, :], k) for i in range(n_paths)])


    def get_for_haplotype(self, haplotype: HaplotypeAsPaths, include_reverse_complements=False):
        """
        Returns all kmers for this haplotype (by using the fact that the haplotype is
        a combination of paths).

        Returns kmers in no particular order.
        """
        kmers_found = []
        # add kmers on first ref node
        start_kmers = self.kmers[int(haplotype.paths[0])][0].ravel()
        print("Staring by adding ", start_kmers)
        encoding = start_kmers.encoding
        kmers_found.append(start_kmers.raw())

        for path in range(len(self.kmers)):
            variants_with_path = np.where(haplotype.paths == path)[0] + 1  # +1 because first element in kmers is the first ref
            kmers = self.kmers[path][variants_with_path].raw().ravel()
            print(path, variants_with_path, bnp.EncodedArray(kmers, encoding))
            kmers_found.append(kmers)

        return bnp.EncodedArray(np.concatenate(kmers_found), encoding)


