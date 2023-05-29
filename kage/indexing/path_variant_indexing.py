import itertools
from dataclasses import dataclass
from typing import List
import npstructures as nps
import numpy as np
import numpy.typing as npt
import bionumpy as bnp
from bionumpy.bnpdataclass import bnpdataclass
import graph_kmer_index
from obgraph.variant_to_nodes import VariantToNodes

"""
Module for simple variant signature finding by using static predetermined paths through the "graph".

Works when all variants are biallelic and there are no overlapping variants.
"""


@dataclass
class GenomeBetweenVariants:
    """ Represents the linear reference genome between variants, not including variant alleles"""
    sequence: bnp.EncodedRaggedArray


class Variants:
    def __init__(self, data: bnp.EncodedRaggedArray, n_alleles: int = 2):
        self._data = data  # data contains sequence for first allele first, then second allele, etc.
        self.n_alleles = n_alleles
        self.n_variants = len(self._data) // n_alleles

    @classmethod
    def from_list(cls, variant_sequences: List[List]):
        zipped = list(itertools.chain(*zip(*variant_sequences)))
        return cls(bnp.as_encoded_array(zipped, bnp.DNAEncoding))

    @property
    def allele_sequences(self):
        return [self._data[self.n_variants*allele:self.n_variants*(allele+1)] for allele in range(self.n_alleles)]

    def get_allele_sequence(self, variant, allele):
        return self._data[self.n_variants*allele + variant].to_string()

    def get_haplotype_sequence(self, haplotypes: np.ndarray) -> bnp.EncodedRaggedArray:
        """
        Gets all sequences from nodes given the haplotypes at those nodes.
        """
        assert len(haplotypes) == self.n_variants
        rows = self.n_variants*haplotypes + np.arange(self.n_variants)
        return self._data[rows]


def zip_sequences(a: bnp.EncodedRaggedArray, b: bnp.EncodedRaggedArray):
    """Utility function for merging encoded ragged arrays ("zipping" rows)"""
    assert len(a) == len(b)+1

    row_lengths = np.zeros(len(a)+len(b))
    row_lengths[0::2] = a.shape[1]
    row_lengths[1::2] = b.shape[1]

    # empty placeholder to fill

    new = bnp.EncodedRaggedArray(
        bnp.EncodedArray(np.zeros(int(np.sum(row_lengths)), dtype=np.uint8), bnp.DNAEncoding),
        row_lengths)
    new[0::2] = a
    new[1:-1:2] = b
    return new


@dataclass
class Graph:
    genome: GenomeBetweenVariants
    variants: Variants

    def n_variants(self):
        return self.variants.n_variants

    def sequence(self, haplotypes: np.ndarray) -> bnp.EncodedArray:
        """
        Returns the sequence through the graph given haplotypes for all variants
        """
        assert len(haplotypes) == self.n_variants()
        ref_sequence = self.genome.sequence
        variant_sequences = self.variants.get_haplotype_sequence(haplotypes)

        # stitch these together
        return zip_sequences(ref_sequence, variant_sequences)

@dataclass
class Paths:
    paths: List[bnp.EncodedRaggedArray]
    # the allele present at each variant in each path
    variant_alleles: np.ndarray
    #path_kmers: List[bnp.EncodedRaggedArray] = None

    def n_variants(self):
        return self.variant_alleles.shape[1]

    def paths_for_allele_at_variant(self, allele, variant):
        relevant_paths = np.where(self.variant_alleles[:, variant] == allele)[0]
        return [self.paths[p] for p in relevant_paths]


    def get_kmers(self, variant, allele, kmer_size=3):
        # gets all kmers of size kmer_size around the variant allele (on all paths relevant for the allele)
        path_sequences = self.paths_for_allele_at_variant(allele, variant)
        window_sequences = []
        for path in path_sequences:
            offset_in_path = path._shape.starts[variant * 2 + 1]
            start = max(0, offset_in_path - kmer_size)
            end = offset_in_path + kmer_size
            window_sequences.append(path.ravel()[start:end].to_string())

        window_sequences = bnp.as_encoded_array(window_sequences, bnp.DNAEncoding)
        kmers = bnp.get_kmers(window_sequences, kmer_size)
        return kmers


class PathCreator:
    def __init__(self, graph, window: int = 3):
        self._graph = graph
        self._variants = graph.variants
        self._genome = graph.genome
        self._window = window

    def run(self):
        alleles = [0, 1]  # possible alleles, assuming biallelic variants
        n_paths = len(alleles)**self._window
        n_variants = self._variants.n_variants
        combinations = self._make_combination_matrix(alleles, n_paths, n_variants)

        paths = []

        # each path will have sequences between each variant and the allele sequences specified in combinations
        ref_between = self._genome.sequence
        for i, alleles in enumerate(combinations):
            # make a new EncodedRaggedArray where every other row is ref/variant
            """"
            n_rows = len(ref_between) + n_variants
            row_lengths = np.zeros(n_rows)
            row_lengths[0::2] = ref_between.shape[1]


            variant_sequences = bnp.as_encoded_array(
                [self._variants.allele_sequences[allele][variant].to_string() for variant, allele in enumerate(alleles)], bnp.DNAEncoding)

            row_lengths[1:-1:2] = variant_sequences.shape[1]

            # empty placeholder to fill
            path = bnp.EncodedRaggedArray(bnp.EncodedArray(np.zeros(int(np.sum(row_lengths)), dtype=np.uint8), bnp.DNAEncoding), row_lengths)
            path[0::2] = ref_between
            path[1:-1:2] = variant_sequences
            """
            path = self._graph.sequence(alleles)
            paths.append(path)

        return Paths(paths, combinations)

    def _make_combination_matrix(self, alleles, n_paths, n_variants):
        # make all possible combinations of variant alleles through the path
        # where all combinations of alleles are present within a window of size self._window
        combinations = itertools.product(*[alleles for _ in range(self._window)])
        # expand combinations to whole genome
        combinations = (itertools.chain(
            *itertools.repeat(c, n_variants // self._window + 1))
            for c in combinations)
        combinations = (itertools.islice(c, n_variants) for c in combinations)
        combination_matrix = np.zeros((n_paths, n_variants), dtype=np.int8)
        for i, c in enumerate(combinations):
            combination_matrix[i] = np.array(list(c))

        return combination_matrix


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



@dataclass
class Signatures:
    ref: nps.RaggedArray
    alt: nps.RaggedArray


class SignatureFinder:
    """
    For each variant chooses a signature (a set of kmers) based one a Scores
    """
    def __init__(self, paths: Paths, scorer: Scorer, k=3):
        self._paths = paths
        self._scorer = scorer
        self._k = k

    def get_as_flat_kmers(self):
        signatures = self.run()

        all_kmers = []
        all_node_ids = []

        for variant_id, (ref_kmers, variant_kmers) in enumerate(zip(signatures.ref, signatures.alt)):
            node_id_ref = variant_id * 2
            node_id_alt = variant_id * 2 + 1
            all_kmers.append(ref_kmers)
            all_node_ids.append(np.full(len(ref_kmers), node_id_ref, dtype=np.uint32))
            all_kmers.append(variant_kmers)
            all_node_ids.append(np.full(len(variant_kmers), node_id_alt, dtype=np.uint32))

        from graph_kmer_index import FlatKmers
        return FlatKmers(np.array(all_kmers), np.array(all_node_ids))

    def run(self) -> Signatures:
        """
        Returns a list of kmers. List contains alleles, each RaggedArray represents the variants
        """
        chosen_ref_kmers = []
        chosen_alt_kmers = []
        encoding = None

        for variant in range(self._paths.n_variants()):
            best_kmers = None
            best_score = -1000000000
            all_ref_kmers = self._paths.get_kmers(variant, 0, self._k)
            all_alt_kmers = self._paths.get_kmers(variant, 1, self._k)
            encoding = all_ref_kmers.encoding

            assert np.all(all_ref_kmers.shape[1] == all_alt_kmers.shape[1])

            for window in range(len(all_ref_kmers[0])):
                ref_kmers = np.unique(all_ref_kmers[:, window].raw())
                alt_kmers = np.unique(all_alt_kmers[:, window].raw())

                all_kmers = np.concatenate([ref_kmers, alt_kmers])
                window_score = self._scorer.score_kmers(all_kmers)

                # if overlap between ref and alt, lower score
                if len(set(ref_kmers).intersection(alt_kmers)) > 0:
                    window_score -= 1000

                if window_score > best_score:
                    best_kmers = [ref_kmers, alt_kmers]
                    best_score = window_score

            chosen_ref_kmers.append(best_kmers[0])
            chosen_alt_kmers.append(best_kmers[1])

        return Signatures(nps.RaggedArray(chosen_ref_kmers), nps.RaggedArray(chosen_alt_kmers))


class SignaturesWithNodes:
    """
    Represents signatures compatible with graph-nodes.
    Nodes are given implicitly from variant ids. Makes it possible
    to create a kmer index that requires nodes.
    """
    pass


class SimpleKmerIndex:
    # Lookup from kmer to variants and alleles
    def __init__(self, lookup):
        self._lookup = lookup

    def get(self, kmers):
        node_ids = lookup[kmers]
        variant_ids = node_ids // 2
        alleles = node_ids % 2
        return variant_ids, alleles

    def get_allele_counts(self, kmers, n_variants, n_alleles=2):
        # returns a matrix of counts over variants. Rows are alleles
        counts = np.zeros((n_alleles, n_variants), dtype=np.uint16)
        variants, alleles = self.get(kmers)
        for variant, allele in zip(variants, alleles):
            counts[allele, variant] += 1

        return counts

    @classmethod
    def from_signatures(cls, signatures: Signatures, modulo=None):
        all_kmers = []
        all_node_ids = []

        for variant_id, (ref_kmers, variant_kmers) in enumerate(zip(signatures.ref, signatures.alt)):
            node_id_ref = variant_id * 2
            node_id_alt = variant_id * 2 + 1
            all_kmers.append(ref_kmers)
            all_node_ids.append(np.full(len(ref_kmers), node_id_ref, dtype=np.uint32))
            all_kmers.append(variant_kmers)
            all_node_ids.append(np.full(len(variant_kmers), node_id_alt, dtype=np.uint32))

        return nps.HashTable(np.concatenate(all_kemrs), np.concatenate(all_node_ids), modulo=modulo)




def index(graph: Graph):
    paths = Paths(graph, window=3)
    variant_signatures = SignatureFinder(paths).get_as_flat_kmers()
    kmer_index = graph_kmer_index.KmerIndex.from_flat_kmers(variant_signatures)
    variant_to_nodes = VariantToNodes(np.arange(graph.n_variants())*2, np.arange(graph.n_variants())*2+1)
