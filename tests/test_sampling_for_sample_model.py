import logging
import pytest
from obgraph import Graph
from obgraph.haplotype_nodes import HaplotypeToNodes
from kage.mapping_model import get_node_counts_from_genotypes, _get_sampled_nodes_and_counts
from kage.sampling_combo_model import RaggedFrequencySamplingComboModel
from graph_kmer_index import KmerIndex, FlatKmers, sequence_to_kmer_hash, CounterKmerIndex
from graph_kmer_index.kmer_counter import KmerCounter
import numpy as np


@pytest.fixture
def graph():
    graph = Graph.from_dicts(
        {1: "ACT", 2: "T", 3: "G", 4: "TGTTTAAA"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 2, 4]
    )
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


def test_get_node_counts_from_genotypes(graph, haplotype_to_nodes, kmer_index):
    k = 3

    logging.info("N haplotypes: %d" % haplotype_to_nodes.n_haplotypes())
    counts = get_node_counts_from_genotypes(haplotype_to_nodes, kmer_index, graph, k=k)
    logging.info(counts)

    assert counts[1][3] == [3]
    assert counts[1][2] == [3]
    assert counts[0][2] == [2]
    assert counts[2][3] == [4]
    assert len(counts[2][2]) == 0

    frequency_model = RaggedFrequencySamplingComboModel.from_counts(counts)

    assert frequency_model.diplotype_counts[2][0][3] == [4]
    assert frequency_model.diplotype_counts[2][1][3] == [1]


def test_get_as_count_matrix(graph, haplotype_to_nodes, kmer_index):
    k = 3
    matrix = _get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, return_matrix_of_counts=True)
    print(matrix)

    assert matrix[1][3][3] == 1
    assert matrix[1][3][2] == 0
