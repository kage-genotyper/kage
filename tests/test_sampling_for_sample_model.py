import logging
import pytest
from obgraph import Graph
from obgraph.haplotype_nodes import HaplotypeToNodes
from kage.mapping_model import get_sampled_nodes_and_counts
from kage.sampling_combo_model import RaggedFrequencySamplingComboModel
from graph_kmer_index import KmerIndex, FlatKmers, sequence_to_kmer_hash, CounterKmerIndex
from graph_kmer_index.kmer_counter import KmerCounter
import numpy as np
np.random.seed(2)


@pytest.fixture
def graph():
    graph = Graph.from_dicts(
        {1: "ACT", 2: "T", 3: "G", 4: "TGTTTAAA"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 2, 4]
    )
    logging.info("Graph chromosome start nodes: %s" % graph.chromosome_start_nodes)
    return graph

@pytest.fixture
def haplotype_to_nodes():
    haplotype_to_nodes = HaplotypeToNodes.from_flat_haplotypes_and_nodes(
        [0, 1, 2, 3], [2, 3, 3, 3]
    )
    return haplotype_to_nodes


@pytest.fixture
def kmer_index():
    kmer_index = CounterKmerIndex.from_kmer_index(KmerIndex.from_flat_kmers(
        FlatKmers(
            np.array([
                sequence_to_kmer_hash("TTT"),
                sequence_to_kmer_hash("TGT")
            ]),
            np.array([
                2, 3
            ])

        )
    ))
    return kmer_index


@pytest.mark.skip
def test_get_as_count_matrix(graph, haplotype_to_nodes, kmer_index):
    k = 3
    matrix = get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, max_count=5)
    assert matrix[1][3][3] == 1
    assert matrix[1][3][2] == 0

    #matrix2 = get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, n_threads=3, max_count=5)
    #assert all([np.all(m1 == m2) for m1, m2 in zip(matrix, matrix2)])


@pytest.mark.skip
def test_parallel(graph, kmer_index):
    k = 3
    n_haplotypes = 100
    # make 100 haplotypes
    haplotype_to_nodes = HaplotypeToNodes.from_flat_haplotypes_and_nodes(
        np.arange(n_haplotypes), np.random.randint(0, 4, n_haplotypes)
    )

    results1 = get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, max_count=8, n_threads=1)
    results10 = get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, max_count=8, n_threads=20)

    assert np.all(results1[0] == results10[0])

