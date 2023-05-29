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
from kmer_mapper.mapper import map_kmers_to_graph_index
from graph_kmer_index import KmerIndex, FlatKmers
from .sparse_haplotype_matrix import SparseHaplotypeMatrix
from ..models.mapping_model import LimitedFrequencySamplingComboModel

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

    def n_nodes(self):
        return self.n_variants()*2

    def sequence(self, haplotypes: np.ndarray) -> bnp.EncodedArray:
        """
        Returns the sequence through the graph given haplotypes for all variants
        """
        assert len(haplotypes) == self.n_variants()
        ref_sequence = self.genome.sequence
        variant_sequences = self.variants.get_haplotype_sequence(haplotypes)

        # stitch these together
        return zip_sequences(ref_sequence, variant_sequences)

    @classmethod
    def from_vcf(cls, vcf_file_name, reference_file_name):
        reference = bnp.open(reference_file_name, bnp.DNAEncoding).read()
        assert len(reference) == 1, "Only one chromosome supported now"

        reference = reference.sequence[0]

        sequences_between_variants = []
        variant_sequences = []

        vcf = bnp.open(vcf_file_name, bnp.DNAEncoding)
        prev_ref_pos = 0
        for chunk in vcf:
            for variant in chunk:
                pos = variant.position
                ref = variant.ref_seq
                alt = variant.alt_seq

                if len(ref) > len(alt) or len(alt) > len(ref):
                    # indel
                    pos_before = pos
                    if len(ref) == 1:
                        # insertion
                        pos_after = pos_before + 1
                        ref = ref[0:0]
                        alt = alt[1:]
                    else:
                        # deletion
                        assert len(alt) == 1
                        pos_after = pos_before + len(ref)
                        alt = alt[0:0]
                        ref = ref[1:]

                else:
                    # snp
                    assert len(ref) == len(alt) == 1
                    pos_before = pos - 1
                    pos_after = pos_before + 2


                sequences_between_variants.append(reference[prev_ref_pos:pos_before+1].to_string())
                prev_ref_pos = pos_after
                variant_sequences.append([ref.to_string(), alt.to_string()])

        # add last bit of reference
        sequences_between_variants.append(reference[prev_ref_pos:].to_string())

        return cls(GenomeBetweenVariants(bnp.as_encoded_array(sequences_between_variants, bnp.DNAEncoding)),
                     Variants.from_list(variant_sequences))


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
    def __init__(self, paths: Paths, scorer: Scorer = None, k=3):
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
        return FlatKmers(np.concatenate(all_kmers), np.concatenate(all_node_ids))

    def get_as_kmer_index(self, include_reverse_complements=True):
        flat = self.get_as_flat_kmers()
        if include_reverse_complements:
            rev_comp_flat = flat.get_reverse_complement_flat_kmers(self._k)
            flat = FlatKmers.from_multiple_flat_kmers([flat, rev_comp_flat])
        index = KmerIndex.from_flat_kmers(flat)
        index.convert_to_int32()
        return index

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
                ref_kmers = np.unique(all_ref_kmers[:, window].raw().ravel())
                alt_kmers = np.unique(all_alt_kmers[:, window].raw().ravel())

                print(ref_kmers, alt_kmers)

                all_kmers = np.concatenate([ref_kmers, alt_kmers])
                window_score = self._scorer.score_kmers(all_kmers) if self._scorer is not None else 0

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


class MappingModelCreator:
    def __init__(self, graph: Graph, kmer_index: KmerIndex,
                 haplotype_matrix: SparseHaplotypeMatrix, max_count=10, k=31):
        self._graph = graph
        self._kmer_index = kmer_index
        self._haplotype_matrix = haplotype_matrix
        self._n_nodes = graph.n_nodes()
        self._counts = LimitedFrequencySamplingComboModel.create_empty(self._n_nodes, max_count)
        self._k = k
        self._max_count = max_count

    def _process_individual(self, i):
        # extract kmers from both haplotypes and map these using the kmer index
        haplotype1 = self._haplotype_matrix.get_haplotype(i*2)
        haplotype2 = self._haplotype_matrix.get_haplotype(i*2+1)

        sequence1 = self._graph.sequence(haplotype1).ravel()
        sequence2 = self._graph.sequence(haplotype2).ravel()

        kmers1 = bnp.get_kmers(sequence1, self._k).ravel().raw().astype(np.uint64)
        kmers2 = bnp.get_kmers(sequence2, self._k).ravel().raw().astype(np.uint64)

        node_counts = self._kmer_index.map_kmers(kmers1, self._n_nodes)
        node_counts += self._kmer_index.map_kmers(kmers2, self._n_nodes)

        # split into nodes that the haplotype has and nodes not
        # mask represents the number of haplotypes this individual has per node (0, 1 or 2 for diploid individuals)
        mask = np.zeros(self._n_nodes, dtype=np.int8)
        mask[haplotype1] += 1
        mask[haplotype2] += 1

        for genotype in [0, 1, 2]:
            nodes_with_genotype = np.where(mask == genotype)[0]
            counts_on_nodes = node_counts[nodes_with_genotype].astype(int)
            below_max_count = np.where(counts_on_nodes < self._max_count)[0]  # ignoring counts larger than supported by matrix
            self._counts.diplotype_counts[genotype][
                nodes_with_genotype[below_max_count], counts_on_nodes[below_max_count]
            ] += 1

    def run(self) -> LimitedFrequencySamplingComboModel:
        n_variants, n_haplotypes = self._haplotype_matrix.shape
        n_nodes = n_variants * 2
        for individual in range(n_haplotypes // 2):
            self._process_individual(individual)

        return self._counts


def index(reference_file_name, vcf_file_name, out_base_name, k=31, variant_window=3):
    graph = Graph.from_vcf(vcf_file_name, reference_file_name)
    haplotype_matrix = SparseHaplotypeMatrix.from_vcf(vcf_file_name)

    paths = PathCreator(graph, window=variant_window).run()
    kmer_index = SignatureFinder(paths, scorer=None).get_as_kmer_index()
    model_creator = MappingModelCreator(graph, kmer_index, haplotype_matrix, k=k)
    count_model = model_creator.run()

    variant_to_nodes = VariantToNodes(np.arange(graph.n_variants())*2, np.arange(graph.n_variants())*2+1)

