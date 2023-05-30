import pytest
from kage.indexing.path_variant_indexing import Paths, Variants, PathCreator, GenomeBetweenVariants, \
    SignatureFinder, SignatureFinder2, zip_sequences, Graph, MappingModelCreator, \
    MatrixVariantWindowKmers, FastApproxCounter, SignatureFinder3
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
import bionumpy as bnp
import numpy as np


@pytest.fixture
def variants():
    #allele_sequences = [
    #    bnp.as_encoded_array(["A", "A", "", "A", "A"], bnp.DNAEncoding),  # ref
    #    bnp.as_encoded_array(["C", "C", "C", "C", "C"], bnp.DNAEncoding),  # alt
    #]
    #return Variants(allele_sequences)
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
    scorer = FastApproxCounter.from_keys_and_values([1, 2, 3], [1, 2, 3], 21)
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
    kmer_index = SignatureFinder(paths).get_as_kmer_index()
    model_creator = MappingModelCreator(graph, kmer_index, haplotype_matrix, k=3)
    model = model_creator.run()

    print(model)


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
