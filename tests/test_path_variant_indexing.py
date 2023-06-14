import logging
logging.basicConfig(level=logging.INFO)
import pytest
from kage.indexing.path_variant_indexing import Paths, VariantAlleleSequences, PathCreator, GenomeBetweenVariants, \
    SignatureFinder, SignatureFinder2, zip_sequences, Graph, MappingModelCreator, \
    MatrixVariantWindowKmers, FastApproxCounter, SignatureFinder3, find_tricky_variants_from_signatures, find_tricky_variants_from_signatures2, \
    Signatures
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
import bionumpy as bnp
import numpy as np
import npstructures as nps
from kage.indexing.path_based_count_model import PathBasedMappingModelCreator
from kage.preprocessing.variants import Variants, VariantPadder


@pytest.fixture
def variants():
    return VariantAlleleSequences.from_list(
        [["A", "C"], ["A", "C"], ["", "C"], ["A", "C"], ["A", "C"]]
    )


@pytest.fixture
def genome():
    return GenomeBetweenVariants(bnp.as_encoded_array(["GGG", "GG", "GG", "GG", "", "GGG"], bnp.DNAEncoding))

@pytest.fixture
def graph(genome, variants):
    return Graph(genome, variants)


@pytest.fixture
def haplotype_matrix(graph):
    return SparseHaplotypeMatrix.from_variants_and_haplotypes(
        np.array([0, 0, 1, 2, 2]), np.array([0, 1, 2, 2, 1]), graph.n_variants(), 4)


def tests_variants(variants):
    assert variants.get_allele_sequence(0, 0) == "A"
    assert variants.get_allele_sequence(0, 1) == "C"
    assert variants.get_allele_sequence(2, 0) == ""


def test_sequence_ragged_array(graph):
    seq = graph.sequence_of_pairs_of_ref_and_variants_as_ragged_array(
        np.array([0, 0, 0, 0, 0])
    )
    assert seq.tolist() == ["GGG", "AGG", "AGG", "GG", "A", "AGGG"]
    seq = graph.sequence_of_pairs_of_ref_and_variants_as_ragged_array(
        np.array([1, 1, 1, 1, 1])
    )
    assert seq.tolist() == ["GGG", "CGG", "CGG", "CGG", "C", "CGGG"]


def test_kmers_from_graph_paths(graph):
    haplotypes = np.array([0, 0, 0, 0, 0])
    kmers = graph.kmers_for_pairs_of_ref_and_variants(haplotypes, k=3)
    kmers = [[k.to_string() for k in node] for node in kmers]
    assert kmers == [
        ["GGG", "GGA", "GAG"],
        ["AGG", "GGA", "GAG"],
        ["AGG", "GGG", "GGG"],
        ["GGA", "GAA"],
        ["AAG"],
        ["AGG", "GGG"]
    ]

    haplotypes = np.array([1, 1, 1, 1, 1])
    kmers = graph.kmers_for_pairs_of_ref_and_variants(haplotypes, k=2)
    kmers = [[k.to_string() for k in node] for node in kmers]
    assert kmers == [
        ["GG", "GG", "GC"],
        ["CG", "GG", "GC"],
        ["CG", "GG", "GC"],
        ["CG", "GG", "GC"],
        ["CC"],
        ["CG", "GG", "GG"]
    ]


def test_get_haplotype_sequence(variants):
    seq = variants.get_haplotype_sequence(np.array([0, 0, 0, 0, 0])).tolist()
    assert seq == ["A", "A", "", "A", "A"]

    seq = variants.get_haplotype_sequence(np.array([0, 1, 1, 0, 1])).tolist()
    assert seq == ["A", "C", "C", "A", "C"]


def test_graph(genome, variants):
    graph = Graph(genome, variants)
    assert graph.sequence(np.array([0, 1, 1, 0, 1])).ravel().to_string() == "GGGAGGCGGCGGACGGG"


def test_create_path(graph):
    creator = PathCreator(graph)
    paths = creator.run()
    print(paths)
    assert len(paths.paths) == 8


def _test_get_kmers(graph):
    creator = PathCreator(graph)
    paths = creator.run()
    kmers = paths.get_kmers(0, 0, kmer_size=4)
    print(kmers)

class DummyScorer:
    def __init__(self):
        pass

    def score_kmers(self, kmers):
        return 1


def test_signatures(graph):
    creator = PathCreator(graph)
    paths = creator.run()
    signature_finder = SignatureFinder2(paths, DummyScorer())
    signatures = signature_finder.run()
    print(signatures)


def test_signature_finder3(graph):
    creator = PathCreator(graph)
    paths = creator.run()
    scorer = FastApproxCounter.from_keys_and_values(np.array([1, 2, 100]), np.array([1, 2, 3]), 21)
    signature_finder = SignatureFinder3(paths, scorer)
    signatures = signature_finder.run()
    print(signatures)


def test_zip_sequences():
    a = bnp.as_encoded_array(["AA", "CC", "GG", "TT", "C"], bnp.DNAEncoding)
    b = bnp.as_encoded_array(["T", "", "T", "AAA"], bnp.DNAEncoding)

    correct = "AATCCGGTTTAAAC"
    assert zip_sequences(a, b).ravel().to_string() == correct



def test_mapping_model_creator(graph, haplotype_matrix):
    paths = PathCreator(graph, window=3).run()
    scorer = FastApproxCounter.from_keys_and_values(np.array([1, 2, 100]), np.array([1, 2, 3]), 21)
    signatures = SignatureFinder3(paths, scorer).run()
    kmer_index = signatures.get_as_kmer_index(k=3)
    model_creator = MappingModelCreator(graph, kmer_index, haplotype_matrix, k=3)
    model = model_creator.run()
    print(model)
    path_based_model = PathBasedMappingModelCreator(graph, kmer_index, haplotype_matrix, window=3, k=3,
                                                    paths_allele_matrix=paths.variant_alleles).run()
    print(path_based_model)
    assert model == path_based_model


def test_graph_from_vcf():
    graph = Graph.from_vcf("example_data/few_variants_two_chromosomes.vcf", "example_data/small_reference_two_chromosomes.fa")

    assert graph.n_variants() == 4
    assert [s for s in graph.genome.sequence.tolist()] == \
        ["AAA", "CC", "CCG", "TTTTAAA", "CC"]

    assert graph.variants.get_allele_sequence(0, 0) == "A"
    assert graph.variants.get_allele_sequence(0, 1) == "T"
    assert graph.variants.get_allele_sequence(1, 0) == ""
    assert graph.variants.get_allele_sequence(1, 1) == "TTT"
    assert graph.variants.get_allele_sequence(2, 0) == "GGG"
    assert graph.variants.get_allele_sequence(2, 1) == ""
    assert graph.variants.get_allele_sequence(3, 0) == "A"
    assert graph.variants.get_allele_sequence(3, 1) == "T"

    # following ref at all variants should give ref sequence
    assert graph.sequence(np.array([0, 0, 0, 0])).ravel().to_string() == "AAAACCCCGGGGTTTTAAAACC"

    # following alt at all variants
    assert graph.sequence(np.array([1, 1, 1, 1])).ravel().to_string() == "AAATCCTTTCCGTTTTAAATCC"


def test_graph_from_padded_overlapping_variants():
    reference = bnp.datatypes.SequenceEntry.from_entry_tuples([("chr1", "CCCCAAAA" + "T"*10)])
    variants = Variants.from_entry_tuples([
        ("chr1", 4, "AAAA", ""),
        ("chr1", 4, "AAAA", "AAGA")
    ])
    graph = Graph.from_variants_and_reference(reference, variants)
    assert [s for s in graph.genome.sequence.tolist()] == \
           ["CCCC", "", "T"*10]

def test_path_windows(graph):
    creator = PathCreator(graph)
    paths = creator.run()
    windows = paths.get_windows_around_variants(3)
    print("WINDOWS")
    print(windows)



def test_matrix_variant_window_kmers(graph):
    creator = PathCreator(graph)
    paths = creator.run()
    kmers = MatrixVariantWindowKmers.from_paths(paths, k=3)
    print("kmers")
    print(kmers)



def test_tricky_variants_from_signatures():
    signatures = Signatures(
        nps.RaggedArray([[1, 2, 3], [4, 5], [50], [123, 1000]]),
        nps.RaggedArray([[1, 2], [100, 200, 1], [51], [4]])
    )
    tricky_variants = find_tricky_variants_from_signatures(signatures)
    assert np.all(tricky_variants.tricky_variants == [True, True, False, True])
    print(tricky_variants)

    tricky_variants = find_tricky_variants_from_signatures2(signatures)
    print(tricky_variants)
    assert np.all(tricky_variants.tricky_variants == [True, False, False, False])


def test_disc_backed_path(graph):
    creator = PathCreator(graph)
    paths = creator.run()
    paths.to_disc_backend("test_paths")
    print(paths.paths[0][0:1])


class DummyScorer2():
    def score_kmers(self, kmers):
        return np.zeros_like(kmers)


@pytest.mark.xfail
def test_signature_finder_with_svs():
    variants = Variants.from_entry_tuples([
            ("chr1", 10, "A", "T"),
            ("chr1", 20, "A", "A"* 10 + "T" + "A"*10)
        ])
    graph = Graph.from_variants_and_reference(
        bnp.datatypes.SequenceEntry.from_entry_tuples([("chr1", "A"*100)]),
        variants
    )
    creator = PathCreator(graph)
    paths = creator.run()
    signature_finder = SignatureFinder3(paths, DummyScorer2(), k=5)
    signatures = signature_finder.run(variants)

    # AAAA kmer should have been replaced for the alt with something else that is unique
    assert 0 not in signatures.alt[1]
    assert len(signatures.alt[1]) >= 1



def test_signatures_with_overlapping_indels():
    variants = Variants.from_entry_tuples([
        ("chr1", 4, "GGGG", ""),
        ("chr1", 6, "G", "T"),
    ])
    sequence = bnp.datatypes.SequenceEntry.from_entry_tuples([("chr1", "ACGTGGGGACGTACGT")])
    padded_variants = VariantPadder(variants, sequence[0].sequence).run()
    print("PAdded variants")
    print(padded_variants)

    print(sequence)
    graph = Graph.from_variants_and_reference(
        sequence, padded_variants
    )
    creator = PathCreator(graph)
    paths = creator.run()
    signature_finder = SignatureFinder3(paths, DummyScorer2(), k=5)
    signatures = signature_finder.run(variants)

    # should be able to find unique signatures for ref
    #assert len(signatures.ref[0]) > 0
    print(signatures)


