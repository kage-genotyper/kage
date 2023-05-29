import pytest
from kage.indexing.path_variant_indexing import Paths, Variants, PathCreator, GenomeBetweenVariants, SignatureFinder, zip_sequences, Graph
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


def test_get_kmers(genome, variants):
    creator = PathCreator(variants, genome)
    paths = creator.run()
    kmers = paths.get_kmers(0, 0)


class DummyScorer:
    def __init__(self):
        pass

    def score_kmers(self, kmers):
        return 1


def test_signatures(graph):
    creator = PathCreator(graph)
    paths = creator.run()
    signature_finder = SignatureFinder(paths, DummyScorer())
    signatures = signature_finder.run()
    print(signatures)


def test_simple_kmer_index():
    index = nps.HashTable()


def test_zip_sequences():
    a = bnp.as_encoded_array(["AA", "CC", "GG", "TT", "C"], bnp.DNAEncoding)
    b = bnp.as_encoded_array(["T", "", "T", "AAA"], bnp.DNAEncoding)

    correct = "AATCCGGTTTAAAC"
    assert zip_sequences(a, b).ravel().to_string() == correct
