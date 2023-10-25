import logging
import pytest
from obgraph import Graph
from obgraph.haplotype_nodes import HaplotypeToNodes
from kage.models.mapping_model import get_sampled_nodes_and_counts
from graph_kmer_index import KmerIndex, FlatKmers, sequence_to_kmer_hash
import numpy as np
from shared_memory_wrapper import free_memory_in_session

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
    kmer_index = KmerIndex.from_flat_kmers(
        FlatKmers(
            np.array([
                sequence_to_kmer_hash("TTT"),
                sequence_to_kmer_hash("TGT")
            ]).astype(np.uint64),
            np.array([
                2, 3
            ])

        )
    )
    kmer_index.convert_to_int32()
    return kmer_index


def test_get_as_count_matrix(graph, haplotype_to_nodes, kmer_index):
    k = 3
    matrix = get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, max_count=5)
    assert matrix[1][3][3] == 1
    assert matrix[1][3][2] == 0


# slow, not necessary
@pytest.mark.skip
def test_parallel(graph, kmer_index):
    k = 3
    n_haplotypes = 100
    # make 100 haplotypes
    haplotype_to_nodes = HaplotypeToNodes.from_flat_haplotypes_and_nodes(
        np.arange(n_haplotypes), np.random.randint(0, 4, n_haplotypes)
    )

    results1 = get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, max_count=8, n_threads=1)
    #print("SHAPE")
    #print(graph.edges._shape.starts)
    results10 = get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, max_count=8, n_threads=20)

    assert np.all(results1[0] == results10[0])
    free_memory_in_session()

