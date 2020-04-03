import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from graph_kmer_index.kmer_index import KmerIndex
from .genotyper import IndependentKmerGenotyper, ReadKmers, BestChainGenotyper

def main():
    run_argument_parser(sys.argv[1:])


def genotype(args):
    logging.info("Reading graphs")
    graph = Graph.from_file(args.graph_file_name)
    sequence_graph = SequenceGraph.from_file(args.graph_file_name + ".sequences")
    kmer_index = KmerIndex.from_file(args.kmer_index)
    linear_path = NumpyIndexedInterval.from_file(args.linear_path_file_name)
    k = args.kmer_size

    read_kmers = ReadKmers.from_fasta_file(args.reads, k)
    logging.info("Starting genotyper")

    logging.info("Method chosen: %s" % args.method)
    if args.method == "all_kmers":
        genotyper = IndependentKmerGenotyper(graph, sequence_graph, linear_path, read_kmers, kmer_index, args.vcf, k)
    elif args.method == "best_chain":
        genotyper = BestChainGenotyper(graph, sequence_graph, linear_path, read_kmers, kmer_index, args.vcf, k)
    else:
        logging.error("Invalid method chosen.")
        return

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
    subparser.set_defaults(func=genotype)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

