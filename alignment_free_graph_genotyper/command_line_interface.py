import sys
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(module)s %(asctime)s %(levelname)s: %(message)s')
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from graph_kmer_index.kmer_index import KmerIndex
from .genotyper import IndependentKmerGenotyper, ReadKmers, BestChainGenotyper, NodeCounts
from numpy_alignments import NumpyAlignments
from graph_kmer_index import ReverseKmerIndex
from .reads import Reads
from .chain_genotyper import ChainGenotyper, CythonChainGenotyper, NumpyNodeCounts


def main():
    run_argument_parser(sys.argv[1:])


def genotype(args):
    logging.info("Reading graphs")
    graph = Graph.from_file(args.graph_file_name)
    sequence_graph = SequenceGraph.from_file(args.graph_file_name + ".sequences")
    kmer_index = KmerIndex.from_file(args.kmer_index)
    linear_path = NumpyIndexedInterval.from_file(args.linear_path_file_name)
    k = args.kmer_size

    logging.info("Initializing readkmers")
    read_kmers = ReadKmers.from_fasta_file(args.reads, k, args.small_k)
    logging.info("Starting genotyper")

    logging.info("Method chosen: %s" % args.method)

    truth_alignments = None
    if args.truth_alignments is not None:
        truth_alignments = NumpyAlignments.from_file(args.truth_alignments)
        logging.info("Read numpy alignments")

    if args.use_node_counts is not None:
        args.write_alignments_to_file = None

    remap_best_chain_to = None
    if args.remap_best_chain_to is not None:
        logging.info("Best chain will be remapped to index: %s" % args.remap_best_chain_to)
        remap_best_chain_to = KmerIndex.from_file(args.remap_best_chain_to)

    align_nodes_to_reads = None
    if args.align_nodes_to_reads is not None:
        logging.info("Variant nodes will be realigned to reads using Reverse Kmer Index")
        align_nodes_to_reads = ReverseKmerIndex.from_file(args.align_nodes_to_reads)


    if args.method == "all_kmers":
        genotyper = IndependentKmerGenotyper(graph, sequence_graph, linear_path, read_kmers, kmer_index, args.vcf, k)
    elif args.method == "best_chain":
        genotyper = BestChainGenotyper(graph, sequence_graph, linear_path, read_kmers, kmer_index, args.vcf, k,
                                       truth_alignments, args.write_alignments_to_file, reference_k=args.small_k,
                                       weight_chains_by_probabilities=args.weight_chains_by_probabilities,
                                       remap_best_chain_to=remap_best_chain_to,
                                       align_nodes_to_reads=align_nodes_to_reads)
    else:
        logging.error("Invalid method chosen.")
        return

    if args.use_node_counts is not None:
        logging.info("Using node counts from file. Not getting counts.")
        genotyper._node_counts = NodeCounts.from_file(args.use_node_counts)
    else:
        genotyper.get_counts()
        if args.node_counts_out_file_name:
            logging.info("Writing node counts to file: %s" % args.node_counts_out_file_name)
            genotyper._node_counts.to_file(args.node_counts_out_file_name)

    genotyper.genotype()


def genotypev2(args):
    logging.info("Reading graphs")
    graph = Graph.from_file(args.graph_file_name)
    sequence_graph = SequenceGraph.from_file(args.graph_file_name + ".sequences")
    kmer_index = KmerIndex.from_file(args.kmer_index)
    linear_path = NumpyIndexedInterval.from_file(args.linear_path_file_name)
    k = args.kmer_size

    reads = Reads.from_fasta(args.reads)

    truth_alignments = None
    if args.truth_alignments is not None:
        truth_alignments = NumpyAlignments.from_file(args.truth_alignments)
        logging.info("Read numpy alignments")

    if args.use_node_counts is not None:
        args.write_alignments_to_file = None

    genotyper_class = ChainGenotyper
    node_counts_class = NodeCounts
    if args.use_cython:
        node_counts_class = NumpyNodeCounts
        reads = args.reads
        genotyper_class = CythonChainGenotyper
        logging.info("Will use the cython genotyper")

    genotyper = genotyper_class(graph, sequence_graph, linear_path, reads, kmer_index, args.vcf, k,
                                       truth_alignments, args.write_alignments_to_file, reference_k=args.small_k)

    if args.use_node_counts is not None:
        logging.info("Using node counts from file. Not getting counts.")
        genotyper._node_counts = node_counts_class.from_file(args.use_node_counts)
    else:
        genotyper.get_counts()
        if args.node_counts_out_file_name:
            logging.info("Writing node counts to file: %s" % args.node_counts_out_file_name)
            genotyper._node_counts.to_file(args.node_counts_out_file_name)

    genotyper.genotype()


# Global numpy arrays used in multithreading, must be defined her to be global
hasher_hashes = None
hashes_to_index = None
n_kmers = None
nodes = None
ref_offsets = None

import numpy as np
from pathos.multiprocessing import Pool
from pathos.pools import ProcessPool
from graph_kmer_index.logn_hash_map import ModuloHashMap
test_array = np.zeros(400000000, dtype=np.int64) + 1

def count_single_thread(reads):
    logging.info("Startin thread")
    hasher = ModuloHashMap(hasher_hashes)
    k = 31
    small_k = 16
    max_node_id = 4000000
    kmer_index = KmerIndex(hasher, hashes_to_index, n_kmers, nodes, ref_offsets)
    logging.info("Got %d lines" % len(reads))
    genotyper = CythonChainGenotyper(None, None, None, reads, kmer_index, None, k, None, None, reference_k=small_k, max_node_id=max_node_id)
    genotyper.get_counts()
    return genotyper._node_counts



def read_chunks(fasta_file_name, chunk_size=10):
    # yields chunks of reads
    file = open(fasta_file_name)
    out = []
    i = 0
    for line in file:
        out.append(line)
        i += 1
        if i >= chunk_size:
            yield out
            out = []
            i = 0



def count(args):
    global hasher_hashes
    global hashes_to_index
    global n_kmers
    global nodes
    global ref_offsets

    logging.info("Reading from file")
    kmer_index = KmerIndex.from_file(args.kmer_index)
    hasher_hashes = kmer_index._hasher._hashes
    hashes_to_index = kmer_index._hashes_to_index
    n_kmers = kmer_index._n_kmers
    nodes = kmer_index._nodes
    ref_offsets = kmer_index._ref_offsets
    reads = read_chunks(args.reads, chunk_size=4000)
    max_node_id = 4000000

    pool = Pool(args.n_threads)
    node_counts = np.zeros(max_node_id+1, dtype=np.int64)
    for result in pool.imap_unordered(count_single_thread, reads):
        print("Got result. Length of counts: %d" % len(result.node_counts))
        node_counts += result.node_counts

    NumpyNodeCounts(node_counts).to_file(args.node_counts_out_file_name)






def run_argument_parser(args):
    parser = argparse.ArgumentParser(
        description='Graph Kmer Index.',
        prog='graph_kmer_index',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("genotype")
    subparser.add_argument("-g", "--graph_file_name", required=True)
    subparser.add_argument("-i", "--kmer_index", required=True)
    subparser.add_argument("-l", "--linear_path_file_name", required=True)
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-v", "--vcf", required=True, help="Vcf to genotype")
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=32)
    subparser.add_argument("-m", "--method", required=False, default='all_kmers')
    subparser.add_argument("-t", "--truth_alignments", required=False)
    subparser.add_argument("-o", "--write_alignments_to_file", required=False)
    subparser.add_argument("-w", "--small-k", required=False, type=int, default=16)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=False)
    subparser.add_argument("-u", "--use_node_counts", required=False)
    subparser.add_argument("-W", "--weight_chains_by_probabilities", required=False, type=bool, default=False)
    subparser.add_argument("-R", "--remap-best-chain-to", required=False, help="Kmerindex that best chains will be remapped to for higher accuracy")
    subparser.add_argument("-L", "--align-nodes-to-reads", required=False, help="Reverse Kmer Index that will be used to check for support of variant nodes in reads")
    subparser.set_defaults(func=genotype)

    subparser = subparsers.add_parser("genotypev2")
    subparser.add_argument("-g", "--graph_file_name", required=True)
    subparser.add_argument("-i", "--kmer_index", required=True)
    subparser.add_argument("-l", "--linear_path_file_name", required=True)
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-v", "--vcf", required=True, help="Vcf to genotype")
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=32)
    subparser.add_argument("-t", "--truth_alignments", required=False)
    subparser.add_argument("-o", "--write_alignments_to_file", required=False)
    subparser.add_argument("-w", "--small-k", required=False, type=int, default=16)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=False)
    subparser.add_argument("-u", "--use_node_counts", required=False)
    subparser.add_argument("-c", "--use_cython", required=False, type=bool, default=False)
    subparser.set_defaults(func=genotypev2)


    subparser = subparsers.add_parser("count")
    subparser.add_argument("-i", "--kmer_index", required=True)
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=32)
    subparser.add_argument("-w", "--small-k", required=False, type=int, default=16)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=False)
    subparser.add_argument("-t", "--n-threads", type=int, default=1, required=False)
    subparser.set_defaults(func=count)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

