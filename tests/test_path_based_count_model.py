import pytest
from kage.indexing.path_variant_indexing import Paths, Graph, Variants, GenomeBetweenVariants, PathCreator
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.indexing.path_based_count_model import HaplotypeAsPaths, PathKmers
import numpy as np
import npstructures as nps
import bionumpy as bnp


@pytest.fixture
def paths():
    return PathCreator.make_combination_matrix(alleles=[0, 1], n_variants=5, window=3)

@pytest.fixture
def paths2():
    return PathCreator.make_combination_matrix(alleles=[0, 1], n_variants=6, window=3)

@pytest.fixture
def graph():
    return Graph(
        GenomeBetweenVariants(bnp.as_encoded_array(["GGG", "GG", "GG", "GG", "GG", "GG"], bnp.DNAEncoding)),
        Variants.from_list([["A", "C"]] * 5)
    )


def test_haplotype_as_paths(paths2):
    # paths:
    # 0: 000000
    # 1: 001001
    # 2: 010010
    # 3: 011011
    # 4: 100100
    # 5: 101101
    # 6: 110110
    # 7: 111111
    haplotype = [0, 1, 0, 1, 1, 0]
    haplotypes_as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles(haplotype, paths2, window=3)
    correct = [2, 6, 6, 6, 6, 6]
    assert np.all(correct == haplotypes_as_paths.paths)

    haplotype = [1, 1, 0, 0, 1, 1]
    haplotypes_as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles(haplotype, paths2, window=3)
    correct = [6, 2, 2, 3, 7, 7]
    assert np.all(correct == haplotypes_as_paths.paths)


def test_path_kmers(graph, paths):
    path_kmers = PathKmers.from_graph_and_paths(graph, paths, k=2)
    ref_path = path_kmers.kmers[0]
    ref_kmers = [[k.to_string() for k in node] for node in ref_path]
    assert ref_kmers == [
        ["GG", "GG", "GA"],
        ["AG", "GG", "GA"],
        ["AG", "GG", "GA"],
        ["AG", "GG", "GA"],
        ["AG", "GG", "GA"],
        ["AG", "GG"]
    ]


def test_get_kmers_for_haplotype(graph, paths):
    path_kmers = PathKmers.from_graph_and_paths(graph, paths, k=2)
    haplotype_as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles([0]*5, paths, window=3)
    kmers = path_kmers.get_for_haplotype(haplotype_as_paths)
    kmers = [k.to_string() for k in kmers]
    correct = ["GG", "GG", "GA", "AG", "GG", "GA", "AG", "GG", "GA", "AG", "GG", "GA", "AG", "GG", "GA", "AG", "GG"]
    assert sorted(kmers) == sorted(correct)


def test_get_kmers_for_haplotype2(graph, paths):
    path_kmers = PathKmers.from_graph_and_paths(graph, paths, k=2)
    haplotype_as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles([0, 1, 0, 1, 0], paths, window=3)
    correct_seq = "GGG A GG C GG A GG C GG A GG".replace(" ", "")
    correct = [correct_seq[i:i+2] for i in range(len(correct_seq)-1)]
    kmers = path_kmers.get_for_haplotype(haplotype_as_paths)
    kmers = [k.to_string() for k in kmers]
    assert sorted(kmers) == sorted(correct)
