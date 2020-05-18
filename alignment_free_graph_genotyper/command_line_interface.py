import gzip
import logging
logging.basicConfig(level=logging.INFO, format='%(module)s %(asctime)s %(levelname)s: %(message)s')
import sys
from pyfaidx import Fasta
import argparse
import time
import os

from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from obgraph import Graph as ObGraph
from graph_kmer_index.kmer_index import KmerIndex
from .genotyper import IndependentKmerGenotyper, ReadKmers, BestChainGenotyper, NodeCounts
from numpy_alignments import NumpyAlignments
from graph_kmer_index import ReverseKmerIndex, CollisionFreeKmerIndex, UniqueKmerIndex
from .reads import Reads
from .chain_genotyper import ChainGenotyper, CythonChainGenotyper, NumpyNodeCounts, UniqueKmerGenotyper

logging.basicConfig(level=logging.INFO, format='%(module)s %(asctime)s %(levelname)s: %(message)s')

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
    #kmer_index = KmerIndex.from_file(args.kmer_index)
    kmer_index = CollisionFreeKmerIndex.from_file(args.kmer_index)
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

    unique_index = None
    if args.unique_index is not None:
        logging.info("Using unique index")
        unique_index = UniqueKmerIndex.from_file(args.unique_index)

    align_nodes_to_reads = None
    if args.align_nodes_to_reads is not None:
        logging.info("Variant nodes will be realigned to reads using Reverse Kmer Index")
        align_nodes_to_reads = ReverseKmerIndex.from_file(args.align_nodes_to_reads)

    genotyper = genotyper_class(graph, sequence_graph, linear_path, reads, kmer_index, args.vcf, k,
                                       truth_alignments, args.write_alignments_to_file, reference_k=args.small_k, max_node_id=4000000,
                                unique_index=unique_index, reverse_index=align_nodes_to_reads)

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
modulo = None
hashes_to_index = None
n_kmers = None
kmers = None
nodes = None
frequencies = None
ref_offsets = None
reference_kmers = None
small_k = None
k = None
graph_edges_indices = None
graph_edges_values = None
graph_edges_n_edges = None
distance_to_node = None
reverse_index_nodes_to_index_positions = None
reverse_index_nodes_to_n_hashes = None
reverse_index_hashes = None
reverse_index_ref_positions = None

import numpy as np
from pathos.multiprocessing import Pool
from pathos.pools import ProcessPool
from graph_kmer_index.logn_hash_map import ModuloHashMap
test_array = np.zeros(400000000, dtype=np.int64) + 1

def count_single_thread(reads):
    logging.info("Startin thread")
    if len(reads) == 0:
        logging.info("Skipping thread, no more reads")
        return None, None

    max_node_id = 160000000
    kmer_index = CollisionFreeKmerIndex(hashes_to_index, n_kmers, nodes, ref_offsets, kmers, modulo, frequencies)
    logging.info("Distance to node type: %s" % distance_to_node.dtype)
    graph = ObGraph(None, None, None, graph_edges_indices, graph_edges_n_edges, graph_edges_values, None, distance_to_node, None)
    reverse_index = ReverseKmerIndex(reverse_index_nodes_to_index_positions, reverse_index_nodes_to_n_hashes, reverse_index_hashes, reverse_index_ref_positions)
    logging.info("Got %d lines" % len(reads))
    genotyper = CythonChainGenotyper(graph, None, None, reads, kmer_index, None, k, None, None, reference_k=small_k, max_node_id=max_node_id, reference_kmers=reference_kmers, reverse_index=reverse_index)
    genotyper.get_counts()
    return genotyper._node_counts, genotyper.chain_positions



def read_chunks(fasta_file_name, chunk_size=10):
    # yields chunks of reads
    file = open(fasta_file_name)
    out = []
    i = 0
    for line in file:
        out.append(line.strip())
        i += 1
        if i >= chunk_size and chunk_size > 0:
            yield out
            out = []
            i = 0
    yield out


def count(args):
    global hashes_to_index
    global n_kmers
    global nodes
    global ref_offsets
    global kmers
    global modulo
    global reference_kmers
    global small_k
    global k
    global frequencies
    global graph_edges_indices
    global graph_edges_values
    global graph_edges_n_edges
    global distance_to_node
    global reverse_index_nodes_to_index_positions
    global reverse_index_nodes_to_n_hashes
    global reverse_index_hashes
    global reverse_index_ref_positions

    truth_positions = None
    if args.truth_alignments is not None:
        truth_alignments = NumpyAlignments.from_file(args.truth_alignments)
        truth_positions = truth_alignments.positions
        logging.info("Read numpy alignments")

    logging.info("Reading from file")
    kmer_index = CollisionFreeKmerIndex.from_file(args.kmer_index)
    hashes_to_index = kmer_index._hashes_to_index
    n_kmers = kmer_index._n_kmers
    nodes = kmer_index._nodes
    ref_offsets = kmer_index._ref_offsets
    kmers = kmer_index._kmers
    modulo = kmer_index._modulo
    small_k = args.small_k
    k = args.kmer_size
    frequencies = kmer_index._frequencies
    graph = ObGraph.from_file(args.graph)
    distance_to_node = graph.ref_offset_to_node

    graph_edges_indices = graph.node_to_edge_index
    graph_edges_values = graph.edges
    graph_edges_n_edges = graph.node_to_n_edges

    reverse_index = ReverseKmerIndex.from_file(args.reverse_index)
    reverse_index_nodes_to_index_positions = reverse_index.nodes_to_index_positions.astype(np.uint32)
    reverse_index_nodes_to_n_hashes = reverse_index.nodes_to_n_hashes.astype(np.uint16)
    reverse_index_hashes = reverse_index.hashes
    reverse_index_ref_positions = reverse_index.ref_positions

    reference_kmers_cache_file_name = "ref.fa.%dmers" % args.small_k
    if os.path.isfile(reference_kmers_cache_file_name + ".npy"):
        logging.info("Used cached reference kmers from file %s.npy" % reference_kmers_cache_file_name)
        reference_kmers = np.load(reference_kmers_cache_file_name + ".npy")
    else:
        logging.info("Creating reference kmers")
        reference_kmers = ReadKmers.get_kmers_from_read_dynamic(str(Fasta("ref.fa")[args.reference_name]), np.power(4, np.arange(0, args.small_k)))
        np.save(reference_kmers_cache_file_name, reference_kmers)

    #reference_kmers = reference_kmers.astype(np.uint64)
    reads = read_chunks(args.reads, chunk_size=args.chunk_size)
    max_node_id = 160000000

    logging.info("Making pool")
    pool = Pool(args.n_threads)
    node_counts = np.zeros(max_node_id+1, dtype=np.int64)
    for result, chain_positions in pool.imap_unordered(count_single_thread, reads):
        if result is not None:
            print("Got result. Length of counts: %d" % len(result.node_counts))
            node_counts += result.node_counts
            if truth_positions is not None:
                n_correct = len(np.where(np.abs(truth_positions - chain_positions) <= 150)[0])
                logging.info("N correct chains: %d" % n_correct)
        else:
            logging.info("No results")

    NumpyNodeCounts(node_counts).to_file(args.node_counts_out_file_name)


def genotype_from_counts(args):
    counts = NumpyNodeCounts.from_file(args.counts)
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


def vcfdiff(args):
    allowed = ["0/0", "0/1", "1/1"]
    assert args.truth_vcf.endswith(".gz"), "Assumes gz for now"
    assert args.genotype in allowed
    truth_positions = set()
    genotype = args.genotype
    truth = gzip.open(args.truth_vcf)
    for i, line in enumerate(truth):
        if i % 1000 == 0:
            logging.info("Reading truth, %d lines" % i)

        line = line.decode("utf-8")
        if line.startswith("#"):
            continue

        l = line.split()
        field = l[9]
        if genotype == "0/1":
            if "0/1" in field or "0|1" in field or "1/0" in field or "1|0" in field:
                truth_positions.add(line.split()[1])
        elif genotype == "1/1":
            if "1/1" in field or "1|1":
                truth_positions.add(line.split()[1])


    vcf = open(args.vcf)
    for line in vcf:
        if line.startswith("#"):
            continue

        if genotype in line:
            pos = line.split()[1]
            if pos not in truth_positions:
                print("False positive: %s" % line.strip())




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
    subparser.add_argument("-U", "--unique_index", required=False, default=None)
    subparser.add_argument("-L", "--align-nodes-to-reads", required=False, help="Reverse Kmer Index that will be used to check for support of variant nodes in reads")
    subparser.set_defaults(func=genotypev2)


    subparser = subparsers.add_parser("count")
    subparser.add_argument("-i", "--kmer_index", required=True)
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=31)
    subparser.add_argument("-w", "--small-k", required=False, type=int, default=16)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=True)
    subparser.add_argument("-t", "--n-threads", type=int, default=1, required=False)
    subparser.add_argument("-c", "--chunk-size", type=int, default=10000, required=False, help="Number of reads to process in the same chunk")
    subparser.add_argument("-T", "--truth_alignments", required=False)
    subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-y", "--reference-name", required=False, default="1")
    subparser.add_argument("-R", "--reverse-index", required=True)
    subparser.set_defaults(func=count)

    subparser = subparsers.add_parser("count_using_unique_index")
    subparser.add_argument("-i", "--unique_kmer_index", required=True)
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=31)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=False)
    subparser.add_argument("-T", "--truth_alignments", required=False)
    subparser.set_defaults(func=count_using_unique_index)


    subparser = subparsers.add_parser("genotype_from_counts")
    subparser.add_argument("-c", "--counts", required=True)
    subparser.add_argument("-g", "--graph_file_name", required=True)
    subparser.add_argument("-v", "--vcf", required=True, help="Vcf to genotype")
    subparser.set_defaults(func=genotype_from_counts)

    subparser = subparsers.add_parser("vcfdiff")
    subparser.add_argument("-v", "--vcf", required=True)
    subparser.add_argument("-t", "--truth-vcf", required=True)
    subparser.add_argument("-g", "--genotype", required=True)
    subparser.set_defaults(func=vcfdiff)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

