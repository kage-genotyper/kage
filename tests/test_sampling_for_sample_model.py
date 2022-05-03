import logging
from obgraph import Graph
from obgraph.haplotype_nodes import HaplotypeToNodes
from kage.mapping_model import get_node_counts_from_genotypes
from kage.sampling_combo_model import RaggedFrequencySamplingComboModel
from graph_kmer_index import KmerIndex, FlatKmers, sequence_to_kmer_hash, CounterKmerIndex
from graph_kmer_index.kmer_counter import KmerCounter
import numpy as np


def test_get_node_counts_from_geontypes():
    k = 3
    graph = Graph.from_dicts(
        {1: "ACT", 2: "T", 3: "G", 4: "TGTTTAAA"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 2, 4]
    )

    haplotype_to_nodes = HaplotypeToNodes.from_flat_haplotypes_and_nodes(
        [0, 1, 2, 3], [2, 3, 3, 3]
    )

    logging.info("N haplotypes: %d" % haplotype_to_nodes.n_haplotypes())

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

    counts = get_node_counts_from_genotypes(haplotype_to_nodes, kmer_index, graph, k=k)
    logging.info(counts)

    assert counts[1][3] == [3]
    assert counts[1][2] == [3]
    assert counts[0][2] == [2]
    assert counts[2][3] == [4]
    assert len(counts[2][2]) == 0

    frequency_model = RaggedFrequencySamplingComboModel.from_counts(1, counts)

    assert frequency_model.diplotype_counts[2][0][3] == [4]
    assert frequency_model.diplotype_counts[2][1][3] == [1]



test_get_node_counts_from_geontypes()
