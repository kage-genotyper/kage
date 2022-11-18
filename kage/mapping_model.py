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
from .util import log_memory_usage_now
from itertools import chain
from graph_kmer_index import KmerIndex
import bionumpy as bnp
#from bionumpy.kmers import KmerEncoding
import math
from npstructures.bitarray import BitArray


def fast_hash_fix(sequence, k, encoding=None):
    hashes = bnp.sequence.get_kmers(
        bnp.EncodedArray(sequence, bnp.DNAEncoding),
        k
    ).ravel().raw().astype(np.uint64)
    return hashes


def _map_haplotype_sequences(sequences, kmer_index, k, n_nodes):
    t = time.perf_counter()
    convolve_time = 0

    if isinstance(kmer_index, KmerIndex):
        counts = np.zeros(n_nodes, dtype=np.uint32)
        from kmer_mapper.mapper import map_kmers_to_graph_index
        for i, sequence in enumerate(sequences):
            # split into even smaller chunks to save memory if sequence is large
            if len(sequence) > 25000000:
                sequence_chunks = np.array_split(sequence, len(sequence)//15000000)
                #logging.info("Split sequence into %d chunks" % len(sequence_chunks))
            else:
                sequence_chunks = [sequence]
                logging.info("did not split sequence")

            for j, s in enumerate(sequence_chunks):
                t2 = time.perf_counter()
                kmers = fast_hash_fix(s, k)
                convolve_time += time.perf_counter()-t2
                t3 = time.perf_counter()
                counts += map_kmers_to_graph_index(kmer_index, n_nodes-1, kmers)

            #log_memory_usage_now("Done mapping sequence %d" % i)
    else:
        kmer_index.reset()
        for i, sequence in enumerate(sequences):
            log_memory_usage_now("Before hashing")
            t2 = time.perf_counter()
            #kmers = np.convolve(sequence, power_vector, mode="valid")
            kmers = fast_hash_fix(sequence.astype(np.uint8), k)
            convolve_time += time.perf_counter()-t2
            log_memory_usage_now("After hashing")

            # split and count to use less memory (Counter.count is very memory-demanding)
            for kmer_chunk in np.array_split(kmers, 8):
                if len(kmer_chunk) == 0:
                    break
                kmer_index.count_kmers(kmer_chunk)

            log_memory_usage_now("After count")

        counts = kmer_index.get_node_counts(n_nodes)

    logging.info("Took %.3f sec to map haplotype sequences for individual" % (time.perf_counter()-t))
    logging.info("Convolve time was %.3f" % convolve_time)
    return counts




def get_sampled_nodes_and_counts(graph, haplotype_to_nodes, k, kmer_index, max_count=30, n_threads=1, limit_to_n_individuals=None):


    n_nodes = len(graph.nodes)

    n_haplotypes = haplotype_to_nodes.n_haplotypes()
    count_matrices = LimitedFrequencySamplingComboModel.create_empty(n_nodes, max_count)
    logging.info("Will process %d haplotypes" % n_haplotypes)

    start_individual = 0
    end_individual = n_haplotypes // 2
    if limit_to_n_individuals is not None:
        end_individual = limit_to_n_individuals
        logging.warning("Will limit to %d individuals" % limit_to_n_individuals)

    logging.info("%d individuals" % end_individual)

    if n_threads == 1:
        count_matrices = _get_sampled_nodes_and_counts_for_range(graph, haplotype_to_nodes, k, kmer_index,
                                            max_count, n_nodes, [start_individual, end_individual])
    else:
        chunks = interval_chunks(0, end_individual, math.ceil(end_individual/n_threads))
        logging.info("Chunks: %s" % chunks)
        count_matrices = parallel_map_reduce_with_adding(_get_sampled_nodes_and_counts_for_range,
                            (graph, haplotype_to_nodes, k, kmer_index, max_count, n_nodes),
                            initial_data=count_matrices,
                            mapper=chunks,
                            n_threads=n_threads
                            )

    return count_matrices



def get_sampled_nodes_and_counts2(graph, haplotype_to_nodes, k, kmer_index, max_count=30, n_threads=16):


    n_nodes = len(graph.nodes)
    n_haplotypes = haplotype_to_nodes.n_haplotypes()
    count_matrices = LimitedFrequencySamplingComboModel.create_empty(n_nodes, max_count)
    logging.info("Will process %d haplotypes" % n_haplotypes)

    start_individual = 0
    end_individual = n_haplotypes // 2
    logging.info("%d individuals" % end_individual)

    from kmer_mapper.mapping import ParalellMapper
    mapper = ParalellMapper(kmer_index, n_nodes, n_threads=n_threads)

    for individual_id in range(0, n_haplotypes//2):
        t_individual = time.perf_counter()
        logging.info("\n\nIndividual %d" % individual_id)
        haplotype_nodes = []
        t = time.perf_counter()
        for haplotype_id in [individual_id * 2, individual_id * 2 + 1]:
            nodes = haplotype_to_nodes.get_nodes(haplotype_id)
            nodes_index = np.zeros(n_nodes, dtype=np.uint8)
            nodes_index[nodes] = 1
            nodes = traverse_graph_by_following_nodes(graph, nodes_index)
            haplotype_nodes.append(nodes)

        logging.info("Took %.3f sec to traverse graph for individual %d" % (time.perf_counter() - t, individual_id))
        log_memory_usage_now("After traversing graph")
        t = time.perf_counter()

        """
        sequences = chain.from_iterable((seq for seq in
                                         graph.get_numeric_node_sequences_by_chromosomes(n))
                                        for n in haplotype_nodes)
        """

        sequences = (graph.get_numeric_node_sequences(n) for n in haplotype_nodes)
        logging.info("Got numeric nnode sequences (%.3f sec)" % (time.perf_counter()-t))

        t_map = time.perf_counter()
        t_hash = 0
        t_node_sequence = time.perf_counter()
        for sequence in sequences:
            logging.info("Took %.3f sec to get node sequences" % (time.perf_counter()-t_node_sequence))
            t0 = time.perf_counter()
            #kmers = kmer_hash_wrapper(sequence.astype(np.int64)[::-1], k)
            t_hash += time.perf_counter()-t0
            #mapper.map(kmers)
            log_memory_usage_now("Before mapping sequence")
            mapper.map_numeric_sequence(sequence[::-1], k)
            log_memory_usage_now("After mapping sequence")
            t_node_sequence = time.perf_counter()

        node_counts = mapper.get_results()
        log_memory_usage_now("After getting mapper results")

        logging.info("Mapping took %.2f sec. Kmer hashing took %.2f sec" % (time.perf_counter()-t, t_hash))
        #logging.info("Sum of node counts: %d" % np.sum(node_counts))

        # split into nodes that the haplotype has and nodes not
        # mask represents the number of haplotypes this individual has per node (0, 1 or 2 for diploid individuals)
        t = time.perf_counter()
        mask = np.zeros(n_nodes, dtype=np.int8)
        mask[haplotype_nodes[0]] += 1
        mask[haplotype_nodes[1]] += 1

        for genotype in [0, 1, 2]:
            nodes_with_genotype = np.where(mask == genotype)[0]
            counts_on_nodes = node_counts[nodes_with_genotype].astype(int)
            # ignoring counts larger than supported by matrix
            below_max_count = np.where(counts_on_nodes < max_count)[0]
            count_matrices.diplotype_counts[genotype][nodes_with_genotype[below_max_count],
                                                      counts_on_nodes[below_max_count]] += 1

        logging.info("Took %.3f sec to update count matrices with counts for individual %d" % (
        time.perf_counter() - t, individual_id))

        mapper.reset()

        logging.info("individual done in %.3f sec" % (time.perf_counter()-t_individual))

    return count_matrices


def _get_sampled_nodes_and_counts_for_range(graph, haplotype_to_nodes, k, kmer_index,
                                            max_count, n_nodes, individual_range):

    log_memory_usage_now("Starting")
    count_matrices = LimitedFrequencySamplingComboModel.create_empty(n_nodes, max_count)
    log_memory_usage_now("Made count matrices")

    start, end = individual_range
    assert end > start
    logging.info("Processing interval %d:%d" % (start, end))

    for individual_id in range(*individual_range):
        t_individual = time.perf_counter()
        log_memory_usage_now("Individual %d" % individual_id)
        logging.info("\n\nIndividual %d" % individual_id)
        haplotype_nodes = []
        t = time.perf_counter()
        for haplotype_id in [individual_id * 2, individual_id * 2 + 1]:
            nodes = haplotype_to_nodes.get_nodes(haplotype_id)
            nodes_index = np.zeros(n_nodes, dtype=np.uint8)
            nodes_index[nodes] = 1
            nodes = traverse_graph_by_following_nodes(graph, nodes_index)
            haplotype_nodes.append(nodes)

        logging.info("Took %.3f sec to traverse graph for individual %d" % (time.perf_counter()-t, individual_id))

        sequences = chain.from_iterable((seq for seq in
                                         graph.get_numeric_node_sequences_by_chromosomes(n))
                     for n in haplotype_nodes)
        log_memory_usage_now("Got sequences")

        node_counts = _map_haplotype_sequences(sequences, kmer_index, k, n_nodes)
        log_memory_usage_now("After mapping")

        # split into nodes that the haplotype has and nodes not
        # mask represents the number of haplotypes this individual has per node (0, 1 or 2 for diploid individuals)
        t = time.perf_counter()
        mask = np.zeros(n_nodes, dtype=np.int8)
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

        logging.info("Took %.3f sec to update count matrices with counts for individual %d" % (time.perf_counter()-t, individual_id))
        logging.info("Took %.3f sec to process individual %d" % (time.perf_counter()-t_individual, individual_id))

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
        power_vector = np.power(4, np.arange(k-1, -1, -1)).astype(np.uint64)
        logging.info("Getting kmers")
        t = time.perf_counter()
        kmers = np.convolve(sequence, power_vector, mode="valid")
        logging.info("Time to get kmers: %.3f" % (time.perf_counter() - t))

        logging.info("N kmers for haplotype: %d" % len(kmers))

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
