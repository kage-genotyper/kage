import pytest
from kage.indexing.path_variant_indexing import Paths, Variants, PathCreator, GenomeBetweenVariants, \
    SignatureFinder, SignatureFinder2, zip_sequences, Graph, MappingModelCreator, \
    MatrixVariantWindowKmers, FastApproxCounter, SignatureFinder3, find_tricky_variants_from_signatures, \
    Signatures
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
import bionumpy as bnp
import numpy as np
import npstructures as nps
from kage.indexing.path_based_count_model import PathBasedMappingModelCreator

@pytest.fixture
def variants():
    return Variants.from_list(
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


def test_get_kmers(graph):
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
    signatures = SignatureFinder(paths).run()
    kmer_index = signatures.get_as_kmer_index(k=3)
    model_creator = MappingModelCreator(graph, kmer_index, haplotype_matrix, k=3)
    model = model_creator.run()
    print(model)
    path_based_model = PathBasedMappingModelCreator(graph, kmer_index, haplotype_matrix, window=3, k=3, paths=paths).run()
    print(path_based_model)
    assert model == path_based_model


def test_graph_from_vcf():
    graph = Graph.from_vcf("example_data/few_variants.vcf", "example_data/small_reference.fa")
    assert graph.n_variants() == 3
    assert [s for s in graph.genome.sequence.tolist()] == \
        ["AAA", "CC", "CCG", "TTTT"]

    assert graph.variants.get_allele_sequence(0, 0) == "A"
    assert graph.variants.get_allele_sequence(0, 1) == "T"
    assert graph.variants.get_allele_sequence(1, 0) == ""
    assert graph.variants.get_allele_sequence(1, 1) == "TTT"
    assert graph.variants.get_allele_sequence(2, 0) == "GGG"
    assert graph.variants.get_allele_sequence(2, 1) == ""

    # following ref at all variants should give ref sequence
    assert graph.sequence(np.array([0, 0, 0])).ravel().to_string() == "AAAACCCCGGGGTTTT"

    # following alt at all variants
    assert graph.sequence(np.array([1, 1, 1])).ravel().to_string() == "AAATCCTTTCCGTTTT"






def test_graph_from_vcf_two_chromosomes():
    graph = Graph.from_vcf("example_data/few_variants_two_chromosomes.vcf", "example_data/small_reference_two_chromosomes.fa")
    assert graph.n_variants() == 4
    print(graph.genome.sequence)
    assert len(graph.genome.sequence) == 5
    assert graph.genome.sequence[3].to_string() == "TTTTAAA"




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