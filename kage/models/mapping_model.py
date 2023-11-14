import logging
import time
import numpy as np
from graph_kmer_index import CollisionFreeKmerIndex
from npstructures.hashtable import HashTable
from obgraph.cython_traversing import traverse_graph_by_following_nodes
from shared_memory_wrapper import get_shared_pool, from_file, close_shared_pool, to_file
from shared_memory_wrapper.util import interval_chunks
from shared_memory_wrapper.util import parallel_map_reduce_with_adding
from .sampling_combo_model import LimitedFrequencySamplingComboModel
from kage.util import log_memory_usage_now
from itertools import chain
import bionumpy as bnp
import math
from kmer_mapper.mapper import map_kmers_to_graph_index
from graph_kmer_index import KmerIndex
from obgraph.haplotype_nodes import DiscBackedHaplotypeToNodes
from obgraph.variant_to_nodes import VariantToNodes


def bnp_get_kmers_wrapper(sequence, k):
    hashes = bnp.sequence.get_kmers(
        bnp.EncodedArray(sequence, bnp.DNAEncoding),
        k
    ).ravel().raw().astype(np.uint64)
    return hashes


def _map_haplotype_sequences(sequences, kmer_index, k, n_nodes):
    t = time.perf_counter()
    convolve_time = 0

    counts = np.zeros(n_nodes+1, dtype=np.uint32)
    for i, sequence in enumerate(sequences):
        # split into even smaller chunks to save memory if sequence is large
        if len(sequence) > 25000000:
            sequence_chunks = np.array_split(sequence, len(sequence)//15000000)
        else:
            sequence_chunks = [sequence]
            logging.info("did not split sequence")

        for j, s in enumerate(sequence_chunks):
            t2 = time.perf_counter()
            kmers = bnp_get_kmers_wrapper(s, k)
            convolve_time += time.perf_counter()-t2
            result = map_kmers_to_graph_index(kmer_index, n_nodes, kmers)
            counts += result

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
        chunks = interval_chunks(0, end_individual, n_threads)
        count_matrices = parallel_map_reduce_with_adding(_get_sampled_nodes_and_counts_for_range,
                            (graph, haplotype_to_nodes, k, kmer_index, max_count, n_nodes),
                            initial_data=count_matrices,
                            mapper=chunks,
                            n_threads=n_threads
                            )

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
            try:
                nodes_index[nodes] = 1
            except IndexError:
                logging.error(f"Haplotype is {haplotype_id}. N nodes is {n_nodes}. Max nodes: {np.max(nodes)}")
                raise
            nodes = traverse_graph_by_following_nodes(graph, nodes_index)
            haplotype_nodes.append(nodes)

        logging.info("Took %.3f sec to traverse graph for individual %d" % (time.perf_counter()-t, individual_id))

        sequences = chain.from_iterable((seq for seq in
                                         graph.get_numeric_node_sequences_by_chromosomes(n))
                     for n in haplotype_nodes)
        log_memory_usage_now("Got sequences")

        node_counts = _map_haplotype_sequences(sequences, kmer_index, k, n_nodes)
        #logging.warning("SKIPPING MAPPING FOR TESTING")
        #node_counts = np.zeros(n_nodes, dtype=np.uint32)
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


def sample_node_counts_from_population_cli(args):
    if args.n_threads > 0:
        logging.info("Creating pool to run in parallel")
        get_shared_pool(args.n_threads)

    logging.info("Reading graph")
    args.graph = from_file(args.graph)
    log_memory_usage_now("After reading graph")

    args.kmer_index = KmerIndex.from_file(args.kmer_index)
    args.kmer_index.convert_to_int32()
    args.kmer_index.remove_ref_offsets()  # not needed, will save us some memory

    log_memory_usage_now("After reading kmer index")

    args.haplotype_to_nodes = DiscBackedHaplotypeToNodes.from_file(args.haplotype_to_nodes)
    log_memory_usage_now("After reading haplotype to nodes")


    limit_to_n_individuals = None
    if args.limit_to_n_individuals > 0:
        limit_to_n_individuals = args.limit_to_n_individuals

    counts = get_sampled_nodes_and_counts(args.graph,
                                          args.haplotype_to_nodes,
                                          args.kmer_size,
                                          args.kmer_index,
                                          max_count=args.max_count,
                                          n_threads=args.n_threads,
                                          limit_to_n_individuals=limit_to_n_individuals
                                          )

    close_shared_pool()
    model = counts  # LimitedFrequencySamplingComboModel(counts)
    to_file(model, args.out_file_name)





def make_sparse_count_model(args):
    model = from_file(args.count_model)
    convert_model_to_sparse(model)
    to_file(model, args.out_file_name)


def convert_model_to_sparse(model):
    for i in range(2):
        model[i] = model[i].as_sparse()


def refine_sampling_model_noncli(model, variant_to_nodes, prior_empty_data=0.1):
    models = [
        model.subset_on_nodes(variant_to_nodes.ref_nodes),
        model.subset_on_nodes(variant_to_nodes.var_nodes)
    ]
    logging.info("Filling missing data")
    for m in models:
        m.fill_empty_data2(prior_empty_data)
        m.astype(np.float16)
    return models

def refine_sampling_model(args):
    model = from_file(args.sampling_model)
    variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
    models = refine_sampling_model_noncli(model, variant_to_nodes)
    to_file(models, args.out_file_name)
    logging.info("Wrote refined model to %s" % args.out_file_name)
