import gzip
import itertools
import logging

from obgraph.genotype_matrix import GenotypeFrequencies

logging.basicConfig(level=logging.INFO, format='%(module)s %(asctime)s %(levelname)s: %(message)s %Y-%m-%d %H:%M:%S')

from itertools import repeat
import sys
from pyfaidx import Fasta
import argparse
import time
import os
from graph_kmer_index.shared_mem import from_shared_memory, to_shared_memory

from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from obgraph import Graph as ObGraph
from graph_kmer_index import KmerIndex
from .genotyper import IndependentKmerGenotyper, ReadKmers, BestChainGenotyper, NodeCounts
from numpy_alignments import NumpyAlignments
from graph_kmer_index import ReverseKmerIndex, CollisionFreeKmerIndex, UniqueKmerIndex
from .reads import Reads
from .chain_genotyper import ChainGenotyper, CythonChainGenotyper, NumpyNodeCounts, UniqueKmerGenotyper
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
                                unique_index=unique_index, reverse_index=align_nodes_to_reads, skip_chaining=args.skip_chaining)

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
max_node_id = None
skip_chaining = False
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

    reference_index = from_shared_memory(ReferenceKmerIndex, "reference_index_shared")

    #kmer_index = CollisionFreeKmerIndex(hashes_to_index, n_kmers, nodes, ref_offsets, kmers, modulo, frequencies)
    kmer_index = from_shared_memory(CollisionFreeKmerIndex, "kmer_index_shared")

    #graph = ObGraph(None, None, None, graph_edges_indices, graph_edges_n_edges, graph_edges_values, None, distance_to_node, None)
    graph = None  #from_shared_memory(ObGraph, "graph_shared")

    #reverse_index = ReverseKmerIndex(reverse_index_nodes_to_index_positions, reverse_index_nodes_to_n_hashes, reverse_index_hashes, reverse_index_ref_positions)
    reverse_index = None  #from_shared_memory(ReverseKmerIndex, "reverse_index_shared")


    logging.info("Got %d lines" % len(reads))
    genotyper = CythonChainGenotyper(graph, None, None, reads, kmer_index, None, k, None, None, reference_k=small_k, max_node_id=max_node_id,
                                     reference_kmers=reference_index, reverse_index=reverse_index, skip_reference_kmers=True, skip_chaining=skip_chaining)
    genotyper.get_counts()
    return genotyper._node_counts, genotyper.chain_positions



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


def run_map_single_thread(reads, args):
    logging.info("Running single thread on %d reads" % len(reads))
    logging.info("Args: %s" % args)
    logging.info("Reading reverse index")
    reverse_index = ReferenceKmerIndex.from_file(args.reverse_index)
    logging.info("Done reading reverse index")
    logging.info("Reading kmerindex from file")
    #index = KmerIndex.from_file(args.kmer_index)

    #index = from_shared_memory(KmerIndex, "kmerindex1")
    index = KmerIndex.from_file(args.kmer_index)
    logging.info("Done reading KmerIndex. Now making CythonKmerIndex.")
    cython_index = CythonKmerIndex(index)
    logging.info("Made CythonKmerIndex")
    k = args.kmer_size
    short_k = args.short_kmer_size
    logging.info("Reading reference kmer index")
    reference_kmers = ReferenceKmerIndex.from_file(args.reference_kmer_index)
    #reference_kmers = CythonReferenceKmerIndex(reference_kmers)
    #ref_index_index = reference_kmers.ref_position_to_index
    ref_index_kmers = reference_kmers.kmers


    from .cython_mapper import map
    positions, counts = map(reads, cython_index, reference_kmers, k, short_k, args.max_node_id, ref_index_kmers, reverse_index)

    truth_positions = None
    if args.truth_alignments is not None:
        truth_alignments = NumpyAlignments.from_file(args.truth_alignments)
        truth_positions = truth_alignments.positions
        logging.info("Read numpy alignments")

    if truth_positions is not None:
        n_correct = len(np.where(np.abs(truth_positions - positions) <= 150)[0])
        n_correct_and_mapped = len(np.where((np.abs(truth_positions - positions) <= 150) & (np.array(positions) > 0))[0])
        logging.info("N correct chains: %d" % n_correct)
        n_reads = len(truth_positions)
        logging.info("Ratio correct among those that were mapped: %.3f" % (n_correct_and_mapped / len(np.where(np.array(positions)[0:n_reads] > 0)[0])))

    return np.array(counts)

def run_map_multiprocess(args):
    reads = read_chunks(args.reads, chunk_size=args.chunk_size)
    max_node_id = args.max_node_id
    logging.info("Making pool with %d workers" % args.n_threads)
    pool = Pool(args.n_threads)
    logging.info("Allocating node counts array")
    node_counts = np.zeros(max_node_id, dtype=np.uint16)
    logging.info("Done allocating")
    index = KmerIndex.from_file(args.kmer_index)
    logging.info("Testing to create a cython kmer index")
    logging.info("Done creating cython kmer index")
    to_shared_memory(index, "kmerindex1")

    n_chunks_in_each_pool = args.n_threads * 3
    data_to_process = zip(reads, repeat(args))
    n_reads_processed = 0
    while True:
        results = pool.starmap(run_map_single_thread, itertools.islice(data_to_process, n_chunks_in_each_pool))
        if results:
            for counts in results:
                n_reads_processed += args.chunk_size
                logging.info("-------- %d reads processed in total ------" % n_reads_processed)
                node_counts += counts
        else:
            logging.info("No results, breaking")
            break
    """
    for counts in pool.starmap(run_map_single_thread, zip(reads, repeat(args))):
        node_counts += counts
    """


    counts = NumpyNodeCounts(node_counts)
    counts.to_file(args.node_counts_out_file_name)
    logging.info("Wrote node counts to file %s" % args.node_counts_out_file_name)



def run_map(args):
    index = KmerIndex.from_file(args.kmer_index)
    cython_index = CythonKmerIndex(index)
    k = args.kmer_size
    short_k = args.short_kmer_size
    reference_kmers = ReferenceKmers(args.reference_fasta_file, args.reference_name, short_k, allow_cache=True)

    with open(args.reads) as f:
        reads = [line.strip() for line in f if not line.startswith(">")]

    positions, node_counts = map(reads, cython_index, reference_kmers, k, short_k, args.max_node_id)

    truth_positions = None
    if args.truth_alignments is not None:
        truth_alignments = NumpyAlignments.from_file(args.truth_alignments)
        truth_positions = truth_alignments.positions
        logging.info("Read numpy alignments")

    if args.node_counts_out_file_name:
        logging.info("Writing node counts to file: %s" % args.node_counts_out_file_name)

        counts = NumpyNodeCounts(node_counts)
        counts.to_file(args.node_counts_out_file_name)

    if truth_positions is not None:
        n_correct = len(np.where(np.abs(truth_positions - positions) <= 150)[0])
        logging.info("N correct chains: %d" % n_correct)


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
    global max_node_id
    global skip_chaining

    truth_positions = None
    if args.truth_alignments is not None:
        truth_alignments = NumpyAlignments.from_file(args.truth_alignments)
        truth_positions = truth_alignments.positions
        logging.info("Read numpy alignments")

    logging.info("Reading from file")
    reference_index = ReferenceKmerIndex.from_file(args.reference_index)
    to_shared_memory(reference_index, "reference_index_shared")


    kmer_index = CollisionFreeKmerIndex.from_file(args.kmer_index)
    to_shared_memory(kmer_index, "kmer_index_shared")

    hashes_to_index = kmer_index._hashes_to_index
    n_kmers = kmer_index._n_kmers
    nodes = kmer_index._nodes
    ref_offsets = kmer_index._ref_offsets
    kmers = kmer_index._kmers
    modulo = kmer_index._modulo
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
    for result, chain_positions in pool.imap_unordered(count_single_thread, reads):
        if result is not None:
            print("Got result. Length of counts: %d" % len(result.node_counts))
            node_counts += result.node_counts
            if truth_positions is not None:
                n_correct = len(np.where(np.abs(truth_positions - chain_positions) <= 150)[0])
                logging.info("N correct chains: %d" % n_correct)
        else:
            logging.info("No results")

    counts = NumpyNodeCounts(node_counts)
    counts.to_file(args.node_counts_out_file_name)


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
        description='Alignment free graph genotyper',
        prog='alignment_free_graph_genotyper',
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
    subparser.add_argument("-n", "--node_counts_out_file_name", required=True)
    subparser.add_argument("-M", "--max_node_id", type=int, default=2000000, required=False)
    subparser.add_argument("-t", "--n-threads", type=int, default=1, required=False)
    subparser.add_argument("-c", "--chunk-size", type=int, default=10000, required=False, help="Number of reads to process in the same chunk")
    subparser.add_argument("-T", "--truth_alignments", required=False)
    #subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-Q", "--reference_index", required=True)
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


    def run_map_using_python(args):
        logging.info("Reading graphs")
        graph = ObGraph.from_file(args.graph)
        ref_kmers = ReferenceKmerIndex.from_file(args.ref_kmers)
        kmer_index = KmerIndex.from_file(args.kmer_index)
        k = args.kmer_size

        reads = Reads.from_fasta(args.reads)

        logging.info("Variant nodes will be realigned to reads using Reverse Kmer Index")
        reverse_kmer_index = ReverseKmerIndex.from_file(args.reverse_index)

        from .chain_mapper import ChainMapper
        mapper = ChainMapper(graph, reads, kmer_index, reverse_kmer_index, k, max_node_id=4000000, max_reads=1000000, linear_reference_kmers=ref_kmers)
        node_counts, positions = mapper.get_counts()
        print(positions)

        node_counts = NumpyNodeCounts(node_counts)
        node_counts.to_file(args.node_counts_out_file_name)
        logging.info("Wrote node counts to file: %s" % args.node_counts_out_file_name)

        truth_positions = None
        if args.truth_alignments is not None:
            truth_alignments = NumpyAlignments.from_file(args.truth_alignments)
            truth_positions = truth_alignments.positions
            logging.info("Read numpy alignments")
            positions = positions[0:len(truth_positions)]
            n_correct = len(np.where(np.abs(truth_positions - positions) <= 150)[0])
            n_correct_and_mapped = len(np.where((np.abs(truth_positions - positions) <= 150) & (np.array(positions) > 0))[0])
            logging.info("N correct chains: %d" % n_correct)
            n_reads = len(truth_positions)
            logging.info("Ratio correct among those that were mapped: %.3f" % (n_correct_and_mapped / len(np.where(np.array(positions)[0:n_reads] > 0)[0])))

    subparser = subparsers.add_parser("map_using_python")
    subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-T", "--truth_alignments", required=False)
    subparser.add_argument("-M", "--max_node_id", type=int, default=2000000, required=False)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=False)
    subparser.add_argument("-R", "--reverse-index", required=True)
    subparser.add_argument("-l", "--ref-kmers", required=True)
    subparser.set_defaults(func=run_map_using_python)


    subparser = subparsers.add_parser("map")
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-s", "--short-kmer-size", type=int, default=16, required=False)
    #subparser.add_argument("-F", "--reference-fasta-file", required=True)
    subparser.add_argument("-F", "--reference-kmer-index", required=True)
    subparser.add_argument("-y", "--reference-name", required=False, default="1")
    subparser.add_argument("-T", "--truth_alignments", required=False)
    subparser.add_argument("-M", "--max_node_id", type=int, default=2000000, required=False)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=False)
    subparser.add_argument("-t", "--n-threads", type=int, default=1, required=False)
    subparser.add_argument("-c", "--chunk-size", type=int, default=10000, required=False, help="Number of reads to process in the same chunk")
    subparser.add_argument("-R", "--reverse-index", required=True)
    subparser.set_defaults(func=run_map_multiprocess)

    def analyse_variants(args):
        from .node_count_model import NodeCountModel
        from obgraph.genotype_matrix import MostSimilarVariantLookup
        most_similar_variants = MostSimilarVariantLookup.from_file(args.most_similar_variants)
        graph = ObGraph.from_file(args.graph_file_name)
        kmer_index = KmerIndex.from_file(args.kmer_index)
        reverse_index = ReverseKmerIndex.from_file(args.reverse_index)
        node_count_model = NodeCountModel.from_file(args.node_count_model)

        analyser = KmerAnalyser(graph, args.kmer_size, GenotypeCalls.from_vcf(args.vcf), kmer_index, reverse_index, GenotypeCalls.from_vcf(args.predicted_vcf),
                                GenotypeCalls.from_vcf(args.truth_vcf), TruthRegions(args.truth_regions_file), NumpyNodeCounts.from_file(args.node_counts),
                                node_count_model, GenotypeFrequencies.from_file(args.genotype_frequencies), most_similar_variants)
        analyser.analyse_unique_kmers_on_variants()

    subparser = subparsers.add_parser("analyse_variants")
    subparser.add_argument("-g", "--graph_file_name", required=True)
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

    def model_kmers_from_haplotype_nodes_single_thread(haplotype, args):
        from .node_count_model import NodeCountModelCreatorFromSimpleChaining
        from obgraph.haplotype_nodes import HaplotypeToNodes
        reference_index = ReferenceKmerIndex.from_file(args.reference_index)
        kmer_index = KmerIndex.from_file(args.kmer_index)
        nodes = HaplotypeToNodes.from_file(args.haplotype_nodes)
        nodes = nodes.get_nodes(haplotype)
        graph = ObGraph.from_file(args.graph_file_name)
        sequence_forward = graph.get_nodes_sequence(nodes)
        #nodes_set = set(nodes[haplotype])
        creator = NodeCountModelCreatorFromSimpleChaining(graph, reference_index, nodes, sequence_forward, kmer_index, args.max_node_id, n_reads_to_simulate=args.n_reads, skip_chaining=args.skip_chaining)
        following, not_following = creator.get_node_counts()
        logging.info("Done with haplotype %d" % haplotype)
        return following, not_following

        """
        expected_node_counts_not_following_node = np.zeros(max_node_id)
        n_individuals_not_following_node = np.zeros(max_node_id)
        expected_node_counts_following_node = np.zeros(max_node_id)
        n_individuals_following_node = np.zeros(max_node_id)

        logging.info("Analysing haplotype %d" % haplotype)
        power_array = get_power_array(args.kmer_size)
        graph = ObGraph.from_file(args.graph_file_name)
        kmer_index = KmerIndex.from_file(args.kmer_index)
        from obgraph.haplotype_nodes import HaplotypeNodes
        nodes = HaplotypeNodes.from_file(args.haplotype_nodes).nodes
        logging.info("Done reading haplotype nodes for haplotype %d" % haplotype)
        #nodes = nodes[haplotype]

        # Increase count for how many individuals follow and do not follow nodes
        n_individuals_following_node[nodes[haplotype]] += 1
        indexes = np.ones(max_node_id)
        indexes[nodes[haplotype]] = 0
        n_individuals_not_following_node[np.nonzero(indexes)] += 1

        nodes_set = set(nodes[haplotype])
        logging.info("Getting forward and reverse haplotype sequence")
        sequence_forward = graph.get_nodes_sequence(nodes[haplotype])
        sequence_reverse = str(Seq(sequence_forward).reverse_complement())
        for sequence in (sequence_forward, sequence_reverse):
            logging.info("Getting read kmers for haplotype %d" % haplotype)
            kmers = read_kmers(sequence, power_array)
            logging.info("Getting kmer hits")
            node_hits = kmer_index.get_nodes_from_multiple_kmers(kmers)

            logging.info("Increasing node counts")
            for node in node_hits:
                if node in nodes_set:
                    expected_node_counts_following_node[node] += 1
                else:
                    expected_node_counts_not_following_node[node] += 1

        return expected_node_counts_following_node, expected_node_counts_not_following_node
        """

    def model_kmers_from_haplotype_nodes(args):
        from obgraph.haplotype_nodes import HaplotypeNodes
        haplotypes = list(range(args.n_haplotypes))

        max_node_id = args.max_node_id
        expected_node_counts_not_following_node = np.zeros(max_node_id+1, dtype=np.float)
        expected_node_counts_following_node = np.zeros(max_node_id+1, dtype=np.float)

        n_chunks_in_each_pool = args.n_threads
        pool = Pool(args.n_threads)
        data_to_process = zip(haplotypes, repeat(args))
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
    subparser.set_defaults(func=model_kmers_from_haplotype_nodes)

    def model_using_kmer_index(args):
        from .node_count_model import NodeCountModelCreatorFromNoChaining, NodeCountModel
        index = KmerIndex.from_file(args.kmer_index)
        graph = ObGraph.from_file(args.graph_file_name)
        reverse_index = ReverseKmerIndex.from_file(args.reverse_node_kmer_index)
        variants = GenotypeCalls.from_vcf(args.vcf)

        model_creator = NodeCountModelCreatorFromNoChaining(index, reverse_index, graph, variants, args.max_node_id)
        counts_following, counts_not_following = model_creator.get_node_counts()
        model = NodeCountModel(counts_following, counts_not_following)
        model.to_file(args.out_file_name)


    subparser = subparsers.add_parser("model_using_kmer_index")
    subparser.add_argument("-g", "--graph_file_name", required=True)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-m", "--max-node-id", type=int, required=True)
    subparser.add_argument("-r", "--reverse_node_kmer_index", required=True)
    subparser.add_argument("-v", "--vcf", required=True)
    subparser.set_defaults(func=model_using_kmer_index)


    def no_chain_map_single_process(reads, args):
        from .no_chain_mapper import NoChainMapper
        #kmer_index = from_shared_memory(KmerIndex, "kmerindex1")
        # more memory usage, faster access
        kmer_index = KmerIndex.from_file(args.kmer_index)
        k = args.kmer_size
        mapper = NoChainMapper(reads, kmer_index, k, args.max_node_id)
        node_counts = mapper.get_counts()
        return node_counts

    def no_chain_map_multiprocess(args):
        reads = read_chunks(args.reads, chunk_size=args.chunk_size)
        max_node_id = args.max_node_id
        logging.info("Making pool with %d workers" % args.n_threads)
        pool = Pool(args.n_threads)
        logging.info("Allocating node counts array")
        node_counts = np.zeros(max_node_id+1, dtype=np.uint16)
        index = KmerIndex.from_file(args.kmer_index)
        to_shared_memory(index, "kmerindex1")

        n_chunks_in_each_pool = args.n_threads
        data_to_process = zip(reads, repeat(args))
        n_reads_processed = 0
        while True:
            results = pool.starmap(no_chain_map_single_process, itertools.islice(data_to_process, n_chunks_in_each_pool))
            if results:
                for counts in results:
                    n_reads_processed += args.chunk_size
                    logging.info("-------- %d reads processed in total ------" % n_reads_processed)
                    node_counts += counts
            else:
                logging.info("No results, breaking")
                break

        node_counts = NumpyNodeCounts(node_counts)
        node_counts.to_file(args.node_counts_out_file_name)
        logging.info("Wrote node counts to file: %s" % args.node_counts_out_file_name)

    def no_chain_map(args):
        from .no_chain_mapper import NoChainMapper
        kmer_index = KmerIndex.from_file(args.kmer_index)
        k = args.kmer_size
        reads = Reads.from_fasta(args.reads)
        mapper = NoChainMapper(reads, kmer_index, k, args.max_node_id)
        node_counts= mapper.get_counts()
        node_counts = NumpyNodeCounts(node_counts)
        node_counts.to_file(args.node_counts_out_file_name)
        logging.info("Wrote node counts to file: %s" % args.node_counts_out_file_name)

    subparser = subparsers.add_parser("no_chain_map")
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-r", "--reads", required=True)
    subparser.add_argument("-M", "--max_node_id", type=int, default=2000000, required=False)
    subparser.add_argument("-n", "--node_counts_out_file_name", required=False)
    subparser.add_argument("-t", "--n-threads", type=int, default=1, required=False)
    subparser.add_argument("-c", "--chunk-size", type=int, default=10000, required=False, help="Number of reads to process in the same chunk")
    subparser.set_defaults(func=no_chain_map_multiprocess)

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
        #variants = GenotypeCalls.from_vcf(args.vcf)
        node_counts = NumpyNodeCounts.from_file(args.counts)
        genotyper = StatisticalNodeCountGenotyper(model, args.vcf, variant_to_nodes, node_counts, genotype_frequencies, most_similar_variant_lookup)
        genotyper.genotype()

    subparser = subparsers.add_parser("statistical_node_count_genotyper")
    subparser.add_argument("-c", "--counts", required=True)
    subparser.add_argument("-g", "--variant-to-nodes", required=True)
    subparser.add_argument("-v", "--vcf", required=True, help="Vcf to genotype")
    subparser.add_argument("-m", "--model", required=False, help="Node count model")
    subparser.add_argument("-G", "--genotype-frequencies", required=True, help="Genotype frequencies")
    subparser.add_argument("-M", "--most_similar_variant_lookup", required=True, help="Most similar variant lookup")
    subparser.set_defaults(func=statistical_node_count_genotyper)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

