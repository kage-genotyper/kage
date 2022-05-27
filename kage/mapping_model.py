import logging
import time
import numpy as np
from graph_kmer_index.nplist import NpList
from npstructures import RaggedArray
from npstructures.hashtable import HashTable
from obgraph.cython_traversing import traverse_graph_by_following_nodes
from shared_memory_wrapper.util import interval_chunks
from shared_memory_wrapper.util import parallel_map_reduce, parallel_map_reduce_with_adding
from .sampling_combo_model import LimitedFrequencySamplingComboModel


def _map_haplotype_sequence(sequence, kmer_index, k, update_counter=True):
    power_vector = np.power(4, np.arange(0, k)).astype(np.uint64)
    kmers = np.convolve(sequence, power_vector, mode="valid")
    kmer_index.count_kmers(kmers, update_counter=update_counter)




def get_node_counts_from_genotypes(
    haplotype_to_nodes, kmer_index, graph, n_haplotypes=None, k=31
):
    """
    Returns three HashTables for gentype 0, 1 and 2 (having node 0, 1 or 2 times)
    Each HashTable has kmer counts for all possible individuals with that genotype
    position/index in that HashTable is node id
    """
    n_nodes = len(graph.nodes)

    sampled_counts, sampled_nodes = get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index,
                                                                 n_haplotypes)
    # sampled_nodes and sampled_counts represent for each possible genotype count (0, 1, 2)
    # nodes and counts. Counts are num

    # put nodes and counts into a ragged array
    output = []
    for nodes, counts in zip(sampled_nodes, sampled_counts):
        nodes = nodes.get_nparray()
        counts = counts.get_nparray()

        data = counts[np.argsort(nodes)]
        lengths = np.bincount(nodes, minlength=n_nodes)
        output.append(RaggedArray(data, lengths))

    return output
    #return [HashTable(nodes.get_nparray(), counts.get_nparray()) for nodes, counts in zip(sampled_nodes, sampled_counts)]



def get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, max_count=30, n_threads=1):

    n_nodes = len(graph.nodes)

    n_haplotypes = haplotype_to_nodes.n_haplotypes()
    count_matrices = LimitedFrequencySamplingComboModel.create_empty(n_nodes, max_count)
    logging.info("Will process %d haplotypes" % n_haplotypes)

    start_individual = 0
    end_individual = n_haplotypes // 2
    logging.info("%d individuals" % end_individual)

    if n_threads == 1:
        count_matrices = _get_sampled_nodes_and_counts_for_range(graph, haplotype_to_nodes, k, kmer_index,
                                            max_count, n_nodes, [start_individual, end_individual])
    else:
        chunks = interval_chunks(0, end_individual, end_individual//n_threads+1)
        logging.info("Chunks: %s" % chunks)
        count_matrices = parallel_map_reduce_with_adding(_get_sampled_nodes_and_counts_for_range,
                            (graph, haplotype_to_nodes, k, kmer_index, max_count, n_nodes),
                            initial_data=count_matrices,
                            mapper=chunks,
                            n_threads=n_threads
                            )

    return count_matrices


def _get_sampled_nodes_and_counts_for_range(graph, haplotype_to_nodes, k, kmer_index,
                                            max_count, n_nodes, individual_range):
    
    count_matrices = LimitedFrequencySamplingComboModel.create_empty(n_nodes, max_count)

    logging.info(str(count_matrices))
    
    start, end = individual_range
    assert end > start
    logging.info("Processing interval %d:%d" % (start, end))

    for individual_id in range(*individual_range):
        logging.info("\n\nIndividual %d" % individual_id)
        haplotype_nodes = []
        for haplotype_id in [individual_id * 2, individual_id * 2 + 1]:
            nodes = haplotype_to_nodes.get_nodes(haplotype_id)
            nodes_index = np.zeros(n_nodes, dtype=np.uint8)
            nodes_index[nodes] = 1
            nodes = traverse_graph_by_following_nodes(graph, nodes_index)
            haplotype_nodes.append(nodes)

            sequence = graph.get_numeric_node_sequences(nodes).astype(np.uint64)
            if len(sequence) == 0:
                logging.info("chromosome start node: %s" % graph.chromosome_start_nodes)
                logging.info("Nodes index: %s" % nodes_index)
                logging.error("Haplotype %d, nodes: %s" % (haplotype_id, nodes))
                logging.error("Haplotype: %d" % haplotype_id)
                raise Exception("Error")

            assert sequence.dtype == np.uint64
            # first time: Create a new counter when counting, next time update that counter
            # reason: don't use same counter as other threads and increase counts for both haplotypes
            update_counter = False
            if haplotype_id == individual_id * 2 + 1:
                update_counter = True
            _map_haplotype_sequence(sequence, kmer_index, k, update_counter=update_counter)

        node_counts = kmer_index.get_node_counts(min_nodes=n_nodes)
        kmer_index.counter.fill(0)  # reset to not keep counts for next mapping

        logging.debug(haplotype_nodes)

        # split into nodes that the haplotype has and nodes not
        # mask represents the number of haplotypes this individual has per node (0, 1 or 2 for diploid individuals)
        mask = np.zeros(n_nodes)
        mask[haplotype_nodes[0]] += 1
        mask[haplotype_nodes[1]] += 1
        for genotype in [0, 1, 2]:
            nodes_with_genotype = np.where(mask == genotype)[0]
            #logging.info("N nodes with genotype %d: %d" % (genotype, len(nodes_with_genotype)))
            counts_on_nodes = node_counts[nodes_with_genotype].astype(int)

            # ignoring counts larger than supported by matrix
            below_max_count = np.where(counts_on_nodes < max_count)[0]
            count_matrices.diplotype_counts[genotype][nodes_with_genotype[below_max_count],
                                     counts_on_nodes[below_max_count]] += 1
            
    return count_matrices


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
        nodes_index[nodes] = 1
        nodes = traverse_graph_by_following_nodes(graph, nodes_index)
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
