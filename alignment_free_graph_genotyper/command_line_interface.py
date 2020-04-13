import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(module)s %(asctime)s %(levelname)s: %(message)s')
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from graph_kmer_index.kmer_index import KmerIndex
from .genotyper import IndependentKmerGenotyper, ReadKmers, BestChainGenotyper, NodeCounts
from numpy_alignments import NumpyAlignments
from graph_kmer_index import ReverseKmerIndex
from .reads import Reads
from .chain_genotyper import ChainGenotyper


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

    genotyper = ChainGenotyper(graph, sequence_graph, linear_path, reads, kmer_index, args.vcf, k,
                                       truth_alignments, args.write_alignments_to_file, reference_k=args.small_k)

    if args.use_node_counts is not None:
        logging.info("Using node counts from file. Not getting counts.")
        genotyper._node_counts = NodeCounts.from_file(args.use_node_counts)
    else:
        genotyper.get_counts()
        if args.node_counts_out_file_name:
            logging.info("Writing node counts to file: %s" % args.node_counts_out_file_name)
            genotyper._node_counts.to_file(args.node_counts_out_file_name)

    genotyper.genotype()





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
    subparser.set_defaults(func=genotypev2)
    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

