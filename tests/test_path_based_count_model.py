import time

import pytest
import ray

from kage.indexing.graph import GenomeBetweenVariants, Graph, MultiAllelicVariantSequences
from kage.preprocessing.variants import VariantAlleleSequences
from kage.indexing.paths import Paths, PathCreator
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.indexing.path_based_count_model import HaplotypeAsPaths, PathKmers, prune_kmers_parallel, \
    get_haplotypes_as_paths, get_haplotypes_as_paths_parallel
import numpy as np
import npstructures as nps
import bionumpy as bnp
from kage.indexing.modulo_filter import ModuloFilter


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
        VariantAlleleSequences.from_list([["A", "C"]] * 5)
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


def test_haplotype_as_paths_multiallelic():
    paths = np.array([
        [0, 1, 0, 1, 0, 1],
        [1, 1, 2, 0, 1, 2],
        [3, 1, 0, 2, 1, 3]
    ])

    haplotype = np.array([0, 1, 2, 0, 1, 3])

    window = 2
    correct = [0, 1, 1, 1, 2, 2]
    haplotype_as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles_multiallelic(haplotype, paths, window=window)
    assert np.all(correct == haplotype_as_paths.paths)


def test_path_kmers(graph, paths):
    path_kmers = PathKmers.from_graph_and_paths(graph, paths, k=2)
    ref_path = list(path_kmers.kmers)[0]
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


class DummyKmerIndex():
    def has_kmers(self, kmers):
        filter_out = [0, 10]
        out = np.array([
            kmer not in filter_out for kmer in kmers
        ])
        return out

    def get_kmers(self):
        return np.array([i for i in range(100) if i not in [0, 10]])


def test_prune_path_kmers(graph, paths):
    path_kmers = PathKmers.from_graph_and_paths(graph, paths, k=2)
    print(path_kmers)
    path_kmers.prune(DummyKmerIndex())
    print(path_kmers)



def test_path_based_count_model_with_multiallelic_variants():
    pass


def test_get_kmers_for_haplotype_multiallelic():
    graph = Graph(
        GenomeBetweenVariants.from_list(["GGG", "GG", "GG", "GG", "GG"]),
        MultiAllelicVariantSequences.from_list([
            ["A", "C"],
            ["A", "C", "T"],
            ["A", "T"],
            ["A", "C"]
        ])
    )

    n_alleles = np.array([2, 3, 2, 2])
    paths = PathCreator.make_combination_matrix_multi_allele(n_alleles, window=4)
    haplotype = np.array([0, 2, 1, 1])
    path_kmers = PathKmers.from_graph_and_paths(graph, paths, k=2)
    haplotype_as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles_multiallelic(
        haplotype, paths, window=2
    )

    kmers = path_kmers.get_for_haplotype(haplotype_as_paths)
    kmers = [k.to_string() for k in kmers]
    correct = ["GG", "GG", "GA", "AG", "GG", "GT", "TG", "GG", "GT", "TG", "GG", "GC", "CG", "GG"]
    assert sorted(kmers) == sorted(correct)


@pytest.mark.skip
def test_prune_many_kmers():
    n_threads = 4
    encoding = bnp.get_kmers(bnp.as_encoded_array("G"*31, bnp.DNAEncoding), 31).encoding
    n = 40_000_000
    n_rows = int(n / 10)
    kmers = bnp.EncodedRaggedArray(
        bnp.EncodedArray(np.random.randint(0, 2**63, n), encoding),
        np.zeros(n_rows, int) + 10
    )
    filter = ModuloFilter(np.random.randint(0, 30, 200_000_033) < 1)
    t0 = time.perf_counter()
    pruned = PathKmers.prune_kmers(kmers, filter)
    print("Time noparallel", time.perf_counter()-t0)

    # parallel
    ray.init(num_cpus=n_threads)
    t0 = time.perf_counter()
    pruned2 = prune_kmers_parallel(kmers, filter, n_threads=n_threads)
    print("Time parallel", time.perf_counter()-t0)

    #assert np.all(pruned2 == pruned)


@pytest.mark.skip
def test_benchmark_haplotype_as_paths_from_haplotype():
    n_variants = 1000000
    n_paths = 128
    haplotype = np.random.randint(0, 2, n_variants)
    paths = np.random.randint(0, 2, (n_paths, n_variants))

    t0 = time.perf_counter()
    haplotype_as_paths = HaplotypeAsPaths.from_haplotype_and_path_alleles(haplotype, paths, window=3)
    print("Time", time.perf_counter()-t0)


def test_get_all_haplotypes_as_paths():
    n_variants = 3000
    n_individuals = 100
    n_paths = 128
    window = 7
    matching_window = 4
    haplotype_matrix = SparseHaplotypeMatrix.from_nonsparse_matrix(
        np.random.randint(0, 2, (n_variants, n_individuals))
    )

    paths = np.random.randint(0, 2, (n_paths, n_variants))

    t0 = time.perf_counter()
    all_haplotypes_as_paths = get_haplotypes_as_paths(haplotype_matrix, paths, matching_window)
    print("Time", time.perf_counter()-t0)


    t0 = time.perf_counter()
    all_haplotypes_as_paths2 = get_haplotypes_as_paths_parallel(haplotype_matrix, paths, matching_window, n_threads=8)
    print("Time", time.perf_counter()-t0)

    for h1, h2 in zip(all_haplotypes_as_paths, all_haplotypes_as_paths2):
        h1 = h1.paths.astype(np.uint8)
        h2 = h2.paths

        mismatch = np.where(h1 != h2)[0]
        #print(mismatch, h1[mismatch], h2[mismatch])
        assert np.all((h1 == h2) | (h1 == 0))  # h1 can be 0 (no match) where h2 finds a match. The new matching matches more

