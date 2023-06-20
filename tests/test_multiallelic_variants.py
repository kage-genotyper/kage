#from kage.preprocessing.variants import MultiAllelelicVariants
import npstructures
import pytest
from bionumpy import Interval
from kage.indexing.graph import Graph, GenomeBetweenVariants
from kage.preprocessing.variants import VariantAlleleSequences, MultiAllelicVariantSequences, Variants, VariantAlleleToNodeMap
from kage.indexing.paths import PathCreator, PathSequences
import bionumpy as bnp
from kage.indexing.signatures import MatrixVariantWindowKmers, MultiAllelicSignatures
import numpy as np
from kage.indexing.paths import PathCombinationMatrix, Paths
from kage.indexing.signatures import MultiAllelicSignatureFinder
import awkward as ak
from kage.indexing.graph import make_multiallelic_graph


@pytest.fixture
def variant_alleles():
    return MultiAllelicVariantSequences.from_list([
        ["A", "C"],
        ["ACTG", "ATTG", ""]
    ])

@pytest.fixture
def genome_between_variants():
    return GenomeBetweenVariants.from_list(["AAAA", "GGGG", "TTTT"])


@pytest.fixture
def graph(genome_between_variants, variant_alleles):
    return Graph(genome_between_variants, variant_alleles)


def test_multi_alleles(variant_alleles):
    sequences = variant_alleles.get_haplotype_sequence([0, 0])
    assert sequences.tolist() == ["A", "ACTG"]
    sequences = variant_alleles.get_haplotype_sequence([1, 2])
    assert sequences.tolist() == ["C", ""]


def test_graph(graph):
    assert graph.sequence([1, 2]).tolist() == ["AAAA", "C", "GGGG", "", "TTTT"]
    assert graph.sequence([0, 0]).tolist() == ["AAAA", "A", "GGGG", "ACTG", "TTTT"]


def test_multiallelic_paths():
    alleles = [2, 3, 2, 4]
    matrix = PathCreator.make_combination_matrix_multi_allele(alleles, window=3)
    print(matrix.matrix)
    assert len(matrix) == 2**3
    assert matrix.shape[1] == 4
    # All alleles should be represented
    for i, allele in enumerate(alleles):
        assert all(a in matrix.matrix[:, i] for a in range(allele))


def test_matrix_window_kmers_flexible_window_size():
    path_sequences = PathSequences([
        bnp.as_encoded_array(["AAA", "CCC", "AAA", "GGG", "AAA"], bnp.DNAEncoding),
        bnp.as_encoded_array(["AAA", "", "CCC", "T", "AAA"], bnp.DNAEncoding),
    ])
    path_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(path_sequences, k=3)

    kmer_encoding = bnp.encodings.kmer_encodings.KmerEncoding(bnp.DNAEncoding, 3)
    kmers = [[[kmer_encoding.to_string(k) for k in kmer] for kmer in path] for path in path_kmers.kmers]

    correct = [
        [["AAC", "ACC", "CCC", "CCA", "CAA"], ["AAG", "AGG", "GGG", "GGA", "GAA"]],
        [["AAC", "ACC"], ["CCT", "CTA", "TAA"]]
    ]

    assert kmers == correct


    variant_alleles = np.array([
        [1, 0],
        [0, 1]
    ])

    assert np.all(path_kmers.get_kmers(0, 1, variant_alleles) == path_kmers.kmers[0][0])


class DummyScorer2():
    def score_kmers(self, kmers):
        return np.zeros_like(kmers)


def test_multiallelic_signature_finder():
    paths = Paths(
        PathSequences.from_list([
            ["AAA", "CCC", "G", "TT", "AAA"],
            ["AAA", "CCC", "G", "TT", "AAA"],
            ["AAA", "G", "G", "CC", "AAA"],
        ]),
        PathCombinationMatrix([
            [0, 1],
            [0, 1],
            [1, 0],
            ]
        )
    )

    signature_finder = MultiAllelicSignatureFinder(paths, k=3, scorer=DummyScorer2())
    signatures = signature_finder.run()

    s = ak.to_list(signatures.signatures)
    assert len(s[0][0]) == 1
    assert len(s[0][1]) == 1
    assert len(s[1][0]) == 1
    assert len(s[1][1]) == 1


def test_multiallelic_signatures_as_kmer_index():
    signatures = MultiAllelicSignatures.from_list([
        [[1], [10, 11], [20]],
        [[5, 6], [5]]
    ])
    node_mapping = VariantAlleleToNodeMap(
        npstructures.RaggedArray([
            [0, 1, 2],
            [3, 4]
        ]),
        biallelic_ref_nodes=np.array([0, 0, 3]),
        biallelic_alt_nodes=np.array([1, 2, 4])
    )

    index = signatures.get_as_kmer_index(3, node_mapping)
    assert np.all(index.get_nodes(1) == [0])
    assert np.all(index.get_nodes(10) == [1])
    assert np.all(index.get_nodes(11) == [1])
    assert np.all(index.get_nodes(5) == [3, 4])


@pytest.fixture
def variants():
    return Variants.from_entry_tuples([
        ("1", 1, "A", "C"),
        ("1", 10, "ACTG", "ATTG"),
        ("1", 10, "ACTG", ""),
        ("2", 10, "A", "C")
    ])


@pytest.fixture
def reference_sequence():
    return bnp.datatypes.SequenceEntry.from_entry_tuples([
        ("1", "CACCCCCCCCACTGCCCC"),
        ("2", "GGGGGGGGGGAGGG")
    ])


def test_get_multiallelic_variant_alleles_from_variants(variants):
    multiallelic, node_ids, intervals = variants.get_multi_allele_variant_alleles()
    assert multiallelic.to_list() == [["A", "C"], ["ACTG", "ATTG", ""], ["A", "C"]]
    assert np.all(node_ids.lookup([0, 0], [0, 1]) == [0, 1])
    assert np.all(node_ids.lookup([1, 1, 1], [0, 1, 2]) == [2, 3, 4])
    assert np.all(node_ids.biallelic_ref_nodes == [0, 2, 2, 5])
    assert np.all(node_ids.biallelic_alt_nodes == [1, 3, 4, 6])

    correct_interval = Interval.from_entry_tuples([
        ("1", 1, 2),
        ("1", 10, 14),
        ("2", 10, 11)
    ])

    assert np.all(intervals == correct_interval)


def test_make_multiallelic_graph(reference_sequence, variants):
    graph, node_mapping = make_multiallelic_graph(reference_sequence, variants)
    assert graph.genome.to_list() == ["C", "CCCCCCCC", "CCCCGGGGGGGGGG", "GGG"]
    assert graph.variants.to_list() == [["A", "C"], ["ACTG", "ATTG", ""], ["A", "C"]]





