import logging
import time

import numpy as np
from npstructures.numpylist import NumpyList
from npstructures.hashtable import HashTable
from obgraph.cython_traversing import traverse_graph_by_following_nodes


def get_node_counts_from_haplotypes(
    haplotype_to_nodes, kmer_index, graph, n_haplotypes=None, k=31
):
    """
    returns four lists
    node ids for nodes followed by haplotypes
    number of kmers mapped to those nodes
    same for not having those nodes in haplotype paths

    every position in the node list corresponds to a count, nodes are not unique
    """

    nodes_having_nodes = NumpyList(dtype=np.int)
    counts_having_nodes = NumpyList(dtype=np.int)
    nodes_not_having_nodes = NumpyList(dtype=np.int)
    counts_not_having_nodes = NumpyList(dtype=np.int)

    if n_haplotypes is None:
        n_haplotypes = haplotype_to_nodes.n_haplotypes()
        logging.info("Will process %d haplotypes" % n_haplotypes)

    nodes_index = np.zeros(len(graph.nodes), dtype=np.uint8)

    for haplotype_id in range(n_haplotypes):
        logging.info("Processing haplotype id %d" % haplotype_id)
        nodes = haplotype_to_nodes.get_nodes(haplotype_id)
        logging.info("Nodes before traversing graph: %d" % len(nodes))
        nodes_index[nodes] = 1
        nodes = traverse_graph_by_following_nodes(graph, nodes_index)
        logging.info("Nodes after traversing graph: %d" % len(nodes))

        logging.info("N nodes for haplotype: %d" % len(nodes))
        t = time.perf_counter()
        sequence = graph.get_numeric_node_sequences(nodes).astype(np.uint64)
        logging.info("Time to get haplotype sequence: %.3f" % (time.perf_counter() - t))
        assert sequence.dtype == np.uint64
        power_vector = np.power(4, np.arange(0, k)).astype(np.uint64)
        logging.info("Getting kmers")
        t = time.perf_counter()
        kmers = np.convolve(sequence, power_vector, mode="valid")
        logging.info("Time to get kmers: %.3f" % (time.perf_counter() - t))

        logging.info("N kmers for haplotype: %d" % len(kmers))
        # complement_sequence = (sequence+2) % 4
        # reverse_kmers = np.consolve(complement_sequence, np.arange(0, k).astype(np.uint64)[::-1])
        # all_kmers = np.concatenate([kmers, reverse_kmers])
        all_kmers = kmers
        # node_counts = map_kmers_to_graph_index(kmer_index, max_node_id, all_kmers)
        t = time.perf_counter()
        kmer_index.count_kmers(all_kmers)
        logging.info("Time to count kmers: %.3f" % (time.perf_counter() - t))
        t = time.perf_counter()
        node_counts = kmer_index.get_node_counts()
        logging.info(
            "Time to get node counts from kmer counts: %.3f" % (time.perf_counter() - t)
        )
        logging.info("Length of node counts: %d" % len(node_counts))

        # split into nodes that the haplotype has and nodes not
        nodes_having_nodes.extend(nodes)
        counts_having_nodes.extend(node_counts[nodes])

        mask_not_following_nodes = np.ones(len(node_counts), dtype=bool)
        mask_not_following_nodes[nodes] = False
        non_nodes = np.nonzero(mask_not_following_nodes)[0]
        nodes_not_having_nodes.extend(non_nodes)
        counts_not_having_nodes.extend(node_counts[non_nodes])

        # reset to not keep counts for next haplotype
        kmer_index.counter.fill(0)

    logging.info("Making hashtable")
    keys = nodes_having_nodes.get_nparray()
    values = counts_having_nodes.get_nparray()

    print(keys)
    print(keys.dtype)
    print(values)
    print(values.dtype)

    return [
        HashTable(
            nodes_having_nodes.get_nparray(),
            counts_having_nodes.get_nparray(),
            mod=kmer_index.counter._mod,
        ),
        HashTable(
            nodes_not_having_nodes.get_nparray(),
            counts_not_having_nodes.get_nparray(),
            mod=kmer_index.counter._mod,
        ),
    ]
