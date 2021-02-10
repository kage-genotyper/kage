import gzip
import itertools
import logging

from obgraph.genotype_matrix import GenotypeFrequencies
from alignment_free_graph_genotyper import cython_chain_genotyper

logging.basicConfig(level=logging.INFO, format='%(module)s %(asctime)s %(levelname)s: %(message)s %Y-%m-%d %H:%M:%S')

from itertools import repeat
import sys
from pyfaidx import Fasta
import argparse
import time
import os
from graph_kmer_index.shared_mem import from_shared_memory, to_shared_memory, remove_shared_memory

from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from obgraph import Graph as ObGraph
from graph_kmer_index import KmerIndex
from .genotyper import IndependentKmerGenotyper, ReadKmers, BestChainGenotyper, NodeCounts
from numpy_alignments import NumpyAlignments
from graph_kmer_index import ReverseKmerIndex, CollisionFreeKmerIndex, UniqueKmerIndex
from .reads import Reads
from .chain_genotyper import ChainGenotyper, CythonChainGenotyper, NodeCounts, UniqueKmerGenotyper
from graph_kmer_index.cython_kmer_index import CythonKmerIndex
from graph_kmer_index.cython_reference_kmer_index import CythonReferenceKmerIndex
from .reference_kmers import ReferenceKmers
from .cython_mapper import map
from graph_kmer_index import ReferenceKmerIndex
from .analysis import KmerAnalyser
from .variants import GenotypeCalls, TruthRegions
from obgraph.haplotype_nodes import HaplotypeNodes
from obgraph.genotype_matrix import GenotypeFrequencies
from obgraph.haplotype_nodes import HaplotypeToNodes

logging.basicConfig(level=logging.INFO, format='%(module)s %(asctime)s %(levelname)s: %(message)s')
import numpy as np
np.random.seed(1)

import platform
logging.info("Using Python version " + platform.python_version())

def main():
    run_argument_parser(sys.argv[1:])


import numpy as np
from pathos.multiprocessing import Pool
from pathos.pools import ProcessPool
from graph_kmer_index.logn_hash_map import ModuloHashMap


def count_single_thread(reads, args):
    logging.info("Startin thread")
    if len(reads) == 0:
        logging.info("Skipping thread, no more reads")
        return None, None

    reference_index = from_shared_memory(ReferenceKmerIndex, "reference_index_shared")

    reference_index_scoring = None
    if args.reference_index_scoring is not None:
        reference_index_scoring = from_shared_memory(ReferenceKmerIndex, "reference_index_scoring_shared")

    kmer_index = from_shared_memory(CollisionFreeKmerIndex, "kmer_index_shared")

    logging.info("Got %d lines" % len(reads))
    start_time = time.time()
    chain_positions, node_counts = cython_chain_genotyper.run(reads,
                                                              kmer_index,
                                                              args.max_node_id,
                                                              args.kmer_size,
                                                              reference_index,
                                                              args.max_index_lookup_frequency,
                                                              0,
                                                              reference_index_scoring)
    logging.info("Time spent on getting node counts: %.5f" % (time.time()-start_time))
    return NodeCounts(node_counts), chain_positions
    #return genotyper._node_counts, genotyper.chain_positions


def read_chunks(fasta_file_name, chunk_size=10):
    logging.info("Read chunks")
    # yields chunks of reads
    file = open(fasta_file_name)
    out = []
    i = 0
    for line in file:
        if line.startswith(">"):
            continue

        if i % 500000 == 0:
            logging.info("Read %d lines" % i)

        out.append(line.strip())
        i += 1
        if i >= chunk_size and chunk_size > 0:
            yield out
            out = []
            i = 0
    yield out


def count(args):
    truth_positions = None
    if args.truth_alignments is not None:
        truth_alignments = NumpyAlignments.from_file(args.truth_alignments)
        truth_positions = truth_alignments.positions
        logging.info("Read numpy alignments")

    logging.info("Reading reference index from file")
    reference_index = ReferenceKmerIndex.from_file(args.reference_index)
    to_shared_memory(reference_index, "reference_index_shared")

    if args.reference_index_scoring is not None:
        reference_index_scoring = ReferenceKmerIndex.from_file(args.reference_index_scoring)
        to_shared_memory(reference_index_scoring, "reference_index_scoring_shared")

    logging.info("Reading kmer index from file")
    kmer_index = CollisionFreeKmerIndex.from_file(args.kmer_index)
    to_shared_memory(kmer_index, "kmer_index_shared")

    k = args.kmer_size
    frequencies = kmer_index._frequencies
    max_node_id = args.max_node_id

    #graph = ObGraph.from_file(args.graph)
    #to_shared_memory(graph, "graph_shared")

    #reverse_index = ReverseKmerIndex.from_file(args.reverse_index)
    #to_shared_memory(reverse_index, "reverse_index_shared")

    #reference_kmers = reference_kmers.astype(np.uint64)
    reads = read_chunks(args.reads, chunk_size=args.chunk_size)

    logging.info("Making pool")
    pool = Pool(args.n_threads)
    node_counts = np.zeros(max_node_id+1, dtype=float)
    for result, chain_positions in pool.starmap(count_single_thread, zip(reads, repeat(args))):
        if result is not None:
            print("Got result. Length of counts: %d" % len(result.node_counts))
            node_counts += result.node_counts
            if truth_positions is not None:
                n_correct = len(np.where(np.abs(truth_positions - chain_positions) <= 150)[0])
                logging.info("N correct chains: %d" % n_correct)
        else:
            logging.info("No results")

    counts = NodeCounts(node_counts)
    counts.to_file(args.node_counts_out_file_name)


def genotype_from_counts(args):
    counts = NodeCounts.from_file(args.counts)
    graph = ObGraph.from_file(args.graph_file_name)
    sequence_graph =  None  #SequenceGraph.from_file(args.graph_file_name + ".sequences")
    linear_path = None  # NumpyIndexedInterval.from_file(args.linear_path_file_name)

    genotyper = CythonChainGenotyper(graph, sequence_graph, linear_path, None, None, args.vcf, None,
                                       None, skip_reference_kmers=True)

    genotyper._node_counts = counts
    genotyper.genotype()


def count_using_unique_index(args):
    unique_index = UniqueKmerIndex.from_file(args.unique_kmer_index)
    k = args.kmer_size
    reads = read_chunks(args.reads, -1).__next__()
    genotyper = UniqueKmerGenotyper(unique_index, reads, k)
    genotyper.get_counts()
    genotyper._node_counts.to_file(args.node_counts_out_file_name)


def run_argument_parser(args):
    parser = argparse.ArgumentParser(
        description='Alignment free graph genotyper',
        prog='alignment_free_graph_genotyper',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("count")
    subparser.add_argument("-i", "--kmer_index", required=True)
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=31)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=True)
    subparser.add_argument("-M", "--max_node_id", type=int, default=2000000, required=False)
    subparser.add_argument("-t", "--n-threads", type=int, default=1, required=False)
    subparser.add_argument("-c", "--chunk-size", type=int, default=10000, required=False, help="Number of reads to process in the same chunk")
    subparser.add_argument("-T", "--truth_alignments", required=False)
    subparser.add_argument("-Q", "--reference_index", required=True)
    subparser.add_argument("-R", "--reference_index_scoring", required=False)
    subparser.add_argument("-I", "--max-index-lookup-frequency", required=False, type=int, default=5)
    subparser.set_defaults(func=count)


    def analyse_variants(args):
        from .node_count_model import NodeCountModel
        from obgraph.genotype_matrix import MostSimilarVariantLookup
        from obgraph.variant_to_nodes import VariantToNodes
        most_similar_variants = MostSimilarVariantLookup.from_file(args.most_similar_variants)
        #graph = ObGraph.from_file(args.graph_file_name)
        variant_nodes = VariantToNodes.from_file(args.variant_nodes)
        kmer_index = KmerIndex.from_file(args.kmer_index)
        reverse_index = ReverseKmerIndex.from_file(args.reverse_index)
        node_count_model = NodeCountModel.from_file(args.node_count_model)

        analyser = KmerAnalyser(variant_nodes, args.kmer_size, GenotypeCalls.from_vcf(args.vcf), kmer_index, reverse_index, GenotypeCalls.from_vcf(args.predicted_vcf),
                                GenotypeCalls.from_vcf(args.truth_vcf), TruthRegions(args.truth_regions_file), NodeCounts.from_file(args.node_counts),
                                node_count_model, GenotypeFrequencies.from_file(args.genotype_frequencies), most_similar_variants)
        analyser.analyse_unique_kmers_on_variants()

    subparser = subparsers.add_parser("analyse_variants")
    subparser.add_argument("-g", "--variant-nodes", required=True)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-R", "--reverse-index", required=True)
    subparser.add_argument("-v", "--vcf", required=True, help="Vcf to genotype")
    subparser.add_argument("-P", "--predicted-vcf", required=True)
    subparser.add_argument("-T", "--truth-vcf", required=True)
    subparser.add_argument("-t", "--truth-regions-file", required=True)
    subparser.add_argument("-n", "--node-counts", required=True)
    subparser.add_argument("-m", "--node-count-model", required=True)
    subparser.add_argument("-G", "--genotype-frequencies", required=True)
    subparser.add_argument("-M", "--most-similar-variants", required=True)
    subparser.set_defaults(func=analyse_variants)

    def model_kmers_from_haplotype_nodes_single_thread(haplotype, random_seed, args):
        from .node_count_model import NodeCountModelCreatorFromSimpleChaining
        from obgraph.haplotype_nodes import HaplotypeToNodes
        #reference_index = ReferenceKmerIndex.from_file(args.reference_index)
        #kmer_index = KmerIndex.from_file(args.kmer_index)
        reference_index = from_shared_memory(ReferenceKmerIndex, "reference_index_shared")
        reference_index_scoring = None
        if args.reference_index_scoring is not None:
            reference_index_scoring = from_shared_memory(ReferenceKmerIndex, "reference_index_scoring_shared")

        kmer_index = from_shared_memory(KmerIndex, "kmer_index_shared")

        nodes = from_shared_memory(HaplotypeToNodes, "haplotype_nodes_shared")
        nodes = nodes.get_nodes(haplotype)
        #graph = ObGraph.from_file(args.graph_file_name)
        graph = from_shared_memory(ObGraph, "graph_shared")

        logging.info("Getting haplotype sequence for haplotype %d" % haplotype)
        time_start = time.time()
        sequence_forward = graph.get_numeric_node_sequences(nodes)
        logging.info("Sequence type: %s" % type(sequence_forward))
        logging.info("Done getting sequence (took %.3f sec)" % (time.time()-time_start))
        #nodes_set = set(nodes[haplotype])
        creator = NodeCountModelCreatorFromSimpleChaining(graph, reference_index, nodes, sequence_forward, kmer_index, args.max_node_id, n_reads_to_simulate=args.n_reads, skip_chaining=args.skip_chaining, max_index_lookup_frequency=args.max_index_lookup_frequency, reference_index_scoring=reference_index_scoring, seed=random_seed)
        following, not_following = creator.get_node_counts()
        logging.info("Done with haplotype %d" % haplotype)
        return following, not_following

    def model_kmers_from_haplotype_nodes(args):
        from obgraph.haplotype_nodes import HaplotypeNodes
        haplotypes = list(range(args.n_haplotypes)) * args.run_n_times
        logging.info("Haplotypes that will be given to jobs and run in parallel: %s" % haplotypes)

        max_node_id = args.max_node_id
        expected_node_counts_not_following_node = np.zeros(max_node_id+1, dtype=np.float)
        expected_node_counts_following_node = np.zeros(max_node_id+1, dtype=np.float)

        logging.info("Reading haplotypenodes")
        nodes = HaplotypeToNodes.from_file(args.haplotype_nodes)
        to_shared_memory(nodes, "haplotype_nodes_shared")

        if args.reference_index_scoring is not None:
            reference_index_scoring = ReferenceKmerIndex.from_file(args.reference_index_scoring)
            to_shared_memory(reference_index_scoring, "reference_index_scoring_shared")

        logging.info("Reading graph")
        graph = ObGraph.from_file(args.graph_file_name)
        to_shared_memory(graph, "graph_shared")

        logging.info("Reading reference index from file")
        reference_index = ReferenceKmerIndex.from_file(args.reference_index)
        to_shared_memory(reference_index, "reference_index_shared")

        logging.info("Reading kmer index from file")
        kmer_index = CollisionFreeKmerIndex.from_file(args.kmer_index)
        to_shared_memory(kmer_index, "kmer_index_shared")

        n_chunks_in_each_pool = args.n_threads
        pool = Pool(args.n_threads)
        random_seeds = list(range(0, len(haplotypes)))
        data_to_process = zip(haplotypes, random_seeds, repeat(args))
        while True:
            results = pool.starmap(model_kmers_from_haplotype_nodes_single_thread, itertools.islice(data_to_process, n_chunks_in_each_pool))
            if results:
                for expected_follow, expected_not_follow in results:
                    expected_node_counts_following_node += expected_follow
                    expected_node_counts_not_following_node += expected_not_follow
            else:
                logging.info("No results, breaking")
                break

        #haplotype_nodes = HaplotypeNodes.from_file(args.haplotype_nodes)
        #n_individuals_following_node = haplotype_nodes.n_haplotypes_on_node
        haplotype_nodes = HaplotypeToNodes.from_file(args.haplotype_nodes)
        logging.info("Counting individuals following nodes")
        n_individuals_following_node = haplotype_nodes.get_n_haplotypes_on_nodes_array(max_node_id+1)
        n_individuals_tot = args.n_haplotypes
        n_individuals_not_following_node = np.zeros(len(n_individuals_following_node)) + n_individuals_tot - n_individuals_following_node


        nonzero = np.where(expected_node_counts_following_node != 0)[0]
        expected_node_counts_following_node[nonzero] = expected_node_counts_following_node[nonzero] / \
                                                       n_individuals_following_node[nonzero]
        nonzero = np.where(expected_node_counts_not_following_node != 0)[0]
        expected_node_counts_not_following_node[nonzero] = expected_node_counts_not_following_node[nonzero] / \
                                                           n_individuals_not_following_node[nonzero]

        assert np.min(expected_node_counts_not_following_node) >= 0
        np.savez(args.out_file_name, node_counts_following_node=expected_node_counts_following_node,
                 node_counts_not_following_node=expected_node_counts_not_following_node)
        logging.info("Wrote expected node counts to file %s" % args.out_file_name)

    subparser = subparsers.add_parser("model")
    subparser.add_argument("-g", "--graph_file_name", required=True)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-t", "--n-threads", type=int, required=True)
    subparser.add_argument("-m", "--max-node-id", type=int, required=True)
    subparser.add_argument("-H", "--haplotype-nodes", required=True)
    subparser.add_argument("-n", "--n-haplotypes", type=int, required=True)
    subparser.add_argument("-N", "--n-reads", type=int, required=True, help="N reads to simulate per genome")
    subparser.add_argument("-s", "--skip-chaining", type=bool, default=False, required=False)
    subparser.add_argument("-Q", "--reference_index", required=True)
    subparser.add_argument("-I", "--max-index-lookup-frequency", required=False, type=int, default=5)
    subparser.add_argument("-T", "--run-n-times", required=False, help="Run the whole simulation N times. Useful when wanting to use more threads than number of haplotypes since multiple haplotypes then can be procesed in parallel.", default=1, type=int)
    subparser.add_argument("-R", "--reference_index_scoring", required=False)
    subparser.set_defaults(func=model_kmers_from_haplotype_nodes)

    def statistical_node_count_genotyper(args):
        from .statistical_node_count_genotyper import StatisticalNodeCountGenotyper
        from .node_count_model import NodeCountModel
        from obgraph.genotype_matrix import MostSimilarVariantLookup, GenotypeFrequencies
        from obgraph.variant_to_nodes import VariantToNodes
        genotype_frequencies = GenotypeFrequencies.from_file(args.genotype_frequencies)
        most_similar_variant_lookup = MostSimilarVariantLookup.from_file(args.most_similar_variant_lookup)
        model = NodeCountModel.from_file(args.model) if args.model is not None else None

        #graph = ObGraph.from_file(args.graph_file_name)
        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)

        variants = GenotypeCalls.from_vcf(args.vcf)
        node_counts = NodeCounts.from_file(args.counts)
        genotyper = StatisticalNodeCountGenotyper(model, variants, variant_to_nodes, node_counts, genotype_frequencies, most_similar_variant_lookup)
        genotyper.genotype()
        variants.to_vcf_file(args.out_file_name)

    subparser = subparsers.add_parser("statistical_node_count_genotyper")
    subparser.add_argument("-c", "--counts", required=True)
    subparser.add_argument("-g", "--variant-to-nodes", required=True)
    subparser.add_argument("-v", "--vcf", required=True, help="Vcf to genotype")
    subparser.add_argument("-m", "--model", required=False, help="Node count model")
    subparser.add_argument("-G", "--genotype-frequencies", required=True, help="Genotype frequencies")
    subparser.add_argument("-M", "--most_similar_variant_lookup", required=True, help="Most similar variant lookup")
    subparser.add_argument("-o", "--out-file-name", required=True, help="Will write genotyped variants to this file")
    subparser.set_defaults(func=statistical_node_count_genotyper)

    def remove_shared_memory_command_line(args):
        remove_shared_memory()
    subparser = subparsers.add_parser("remove_shared_memory")
    subparser.set_defaults(func=remove_shared_memory_command_line)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

