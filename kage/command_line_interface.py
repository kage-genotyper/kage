import logging
import sys
import platform
from .util import vcf_pl_and_gl_header_lines, convert_string_genotypes_to_numeric_array, _write_genotype_debug_data, \
    log_memory_usage_now

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
import argparse, time, random
from kage.modles.helper_index import create_helper_model
from .configuration import GenotypingConfig
from kage.models.mapping_model import sample_node_counts_from_population_cli, refine_sampling_model
from shared_memory_wrapper import (
    remove_shared_memory_in_session,
    get_shared_pool,
    close_shared_pool, from_file,
)
from shared_memory_wrapper.shared_memory import remove_all_shared_memory
from graph_kmer_index import KmerIndex, ReverseKmerIndex
from kage.analysis.analysis import analyse_variants
import numpy as np
from .node_counts import NodeCounts
from obgraph.variant_to_nodes import VariantToNodes
from .indexing.tricky_variants import find_variants_with_nonunique_kmers, find_tricky_variants
from .indexing.index_bundle import IndexBundle
from .genotyping.combination_model_genotyper import CombinationModelGenotyper
from kmer_mapper.command_line_interface import map_bnp
from argparse import Namespace

np.random.seed(1)
np.seterr(all="ignore")
np.set_printoptions(suppress=True)


def main():
    run_argument_parser(sys.argv[1:])


def get_kmer_counts(kmer_index, k, reads_file_name, n_threads, gpu=False):
    logging.info("Will count kmers.")
    # temp hack to call kmer_mapper by using the command line interface
    return NodeCounts(map_bnp(Namespace(
        kmer_size=k, kmer_index=kmer_index, reads=reads_file_name, n_threads=n_threads, gpu=gpu, debug=False,
        chunk_size=10000000, map_reverse_complements=True if gpu else False, func=None, output_file=None
    )))


def genotype(args):
    start_time = time.perf_counter()
    logging.info("Read coverage is set to %.3f" % args.average_coverage)
    get_shared_pool(args.n_threads)

    logging.info("Reading all indexes from an index bundle")
    index = IndexBundle.from_file(args.index_bundle, skip=["KmerIndex"]).indexes
    config = GenotypingConfig.from_command_line_args(args)

    if args.counts is None:
        kmer_index = index.kmer_index
        if args.kmer_index is not None:
            logging.info("Not using index from index bundle, but instead using %s" % args.kmer_index)
            kmer_index = KmerIndex.from_file(args.kmer_index)
        # map with kmer_mapper to get node counts
        assert args.reads is not None, "--reads must be specified if not node_counts is specified"
        node_counts = get_kmer_counts(kmer_index, args.kmer_size, args.reads, config.n_threads, args.gpu)
    else:
        node_counts = NodeCounts.from_file(args.counts)

    max_variant_id = len(index.variant_to_nodes.ref_nodes) - 1
    logging.info("Max variant id is assumed to be %d" % max_variant_id)
    log_memory_usage_now("After reading index and node counts")

    genotyper = CombinationModelGenotyper(0, max_variant_id, node_counts, index, config=config)
    log_memory_usage_now("After creating Genotyper ")
    genotypes, probs, count_probs = genotyper.genotype()
    log_memory_usage_now("After genotyping")

    numpy_genotypes = convert_string_genotypes_to_numeric_array(genotypes)
    index.numpy_variants.to_vcf_with_genotypes(
        args.out_file_name,
        config.sample_name_output,
        numpy_genotypes,
        add_header_lines=vcf_pl_and_gl_header_lines(),
        ignore_homo_ref=config.ignore_homo_ref,
        add_genotype_likelyhoods=probs if not config.do_not_write_genotype_likelihoods else None,
    )

    close_shared_pool()
    logging.info("Genotyping took %d sec" % (time.perf_counter() - start_time))
    _write_genotype_debug_data(genotypes, numpy_genotypes, args.out_file_name, index.variant_to_nodes, probs, count_probs)


def run_argument_parser(args):
    parser = argparse.ArgumentParser(
        description="kage",
        prog="kage",
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=50, width=100
        ),
    )

    subparsers = parser.add_subparsers()


    subparser = subparsers.add_parser("genotype")
    subparser.add_argument("-c", "--counts", required=False)
    subparser.add_argument("-r", "--reads", required=False)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=31)
    subparser.add_argument("-g", "--gpu", required=False, type=bool, default=False)
    subparser.add_argument("-i", "--index-bundle", required=True)
    subparser.add_argument("-m", "--kmer-index", required=False, help="Can be specified to override kmer index in index bundle for mapping.")
    subparser.add_argument("-v", "--vcf", required=False, help="Vcf to genotype")
    subparser.add_argument("-o", "--out-file-name", required=True, help="Will write genotyped variants to this file")
    subparser.add_argument("-t", "--n-threads", type=int, required=False, default=8)
    subparser.add_argument("-a", "--average-coverage", type=float, default=15, help="Expected average read coverage", )
    subparser.add_argument("-q", "--min-genotype-quality", type=float, default=0.0,
        help="Min prob of genotype being correct. Genotypes with prob less than this are set to homo ref.")
    subparser.add_argument( "-s", "--sample-name-output", required=False, default="DONOR", help="Sample name that will be used in the output vcf")
    subparser.add_argument( "-u", "--use-naive-priors", required=False, type=bool, default=False,
        help="Set to True to use only population allele frequencies as priors.")
    subparser.add_argument("-I", "--ignore-helper-model", required=False, type=bool, default=False)
    subparser.add_argument("-V", "--ignore-helper-variants", required=False, type=bool, default=False)
    subparser.add_argument("-b", "--ignore-homo-ref", required=False, type=bool, default=False, help="Set to True to not write homo ref variants to output vcf")
    subparser.add_argument("-B", "--do-not-write-genotype-likelihoods", required=False, type=bool, default=False, help="Set to True to not write genotype likelihoods to output vcf")
    subparser.set_defaults(func=genotype)

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
    subparser.add_argument("-m", "--model", required=True)
    subparser.add_argument("-f", "--helper-variants", required=True)
    subparser.add_argument("-F", "--combination-matrix", required=True)
    subparser.add_argument("-p", "--probs", required=True)
    subparser.add_argument("-c", "--count_probs", required=True)
    subparser.set_defaults(func=analyse_variants)

    def run_tests(args):
        from kage.simulation.simulation import run_genotyper_on_simualated_data

        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        genotyper = globals()[args.genotyper]

        run_genotyper_on_simualated_data(
            genotyper,
            args.n_variants,
            args.n_individuals,
            args.average_coverage,
            args.coverage_std,
            args.duplication_rate,
        )

    subparser = subparsers.add_parser("test")
    subparser.add_argument(
        "-g",
        "--genotyper",
        required=False,
        default="CombinationModelGenotyper",
        help="Classname of genotyper",
    )
    subparser.add_argument(
        "-n",
        "--n_variants",
        required=False,
        type=int,
        default=100,
        help="Number of variants to test on",
    )
    subparser.add_argument(
        "-i",
        "--n_individuals",
        required=False,
        type=int,
        default=50,
        help="Number of individuals",
    )
    subparser.add_argument(
        "-r", "--random_seed", required=False, type=int, default=1, help="Random seed"
    )
    subparser.add_argument(
        "-c",
        "--average_coverage",
        required=False,
        type=int,
        default=8,
        help="Average coverage",
    )
    subparser.add_argument(
        "-s", "--coverage_std", required=False, type=int, default=2, help="Coverage std"
    )
    subparser.add_argument(
        "-d",
        "--duplication_rate",
        required=False,
        type=float,
        default=0.1,
        help="Ratio of variants with duplications",
    )
    subparser.set_defaults(func=run_tests)


    subparser = subparsers.add_parser("find_tricky_variants")
    subparser.add_argument("-v", "--variant-to-nodes", required=True)
    subparser.add_argument("-m", "--node-count-model", required=True)
    subparser.add_argument("-r", "--reverse-kmer-index", required=True)
    subparser.add_argument(
        "-M",
        "--max-counts-model",
        required=False,
        type=int,
        default=3,
        help="If model count exceeds this number, variant is tricky",
    )
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument(
        "-u",
        "--only-allow-unique",
        required=False,
        type=bool,
        help="Only allow variants where all kmers are unique",
    )
    subparser.set_defaults(func=find_tricky_variants)


    subparser = subparsers.add_parser("find_variants_with_nonunique_kmers")
    subparser.add_argument("-v", "--variant-to-nodes", required=True, type=VariantToNodes.from_file)
    subparser.add_argument("-r", "--reverse-kmer-index", required=True, type=ReverseKmerIndex.from_file)
    subparser.add_argument("-i", "--population-kmer-index", required=True, type=KmerIndex.from_file)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=find_variants_with_nonunique_kmers)


    def remove_shared_memory_command_line(args):
        remove_all_shared_memory()

    subparser = subparsers.add_parser("free_memory")
    subparser.set_defaults(func=remove_shared_memory_command_line)

    def filter_variants(args):
        from obgraph.variants import VcfVariant

        f = open(args.vcf)
        n_snps_filtered = 0
        n_indels_filtered = 0
        for line in f:
            if line.startswith("#"):
                print(line.strip())
                continue

            variant = VcfVariant.from_vcf_line(line)
            if args.skip_snps and variant.type == "SNP":
                n_snps_filtered += 1
                continue

            if variant.type == "DELETION" or variant.type == "INSERTION":
                if variant.length() < args.minimum_indel_length:
                    n_indels_filtered += 1
                    continue

            print(line.strip())

        logging.info("%d snps filtered" % n_snps_filtered)
        logging.info("%d indels filtered" % n_indels_filtered)

    subparser = subparsers.add_parser("filter_variants")
    subparser.add_argument("-v", "--vcf", required=True, help="Vcf to filter")
    subparser.add_argument(
        "-l", "--minimum-indel-length", required=False, type=int, default=0
    )
    subparser.add_argument(
        "-s", "--skip-snps", required=False, type=bool, default=False
    )
    subparser.set_defaults(func=filter_variants)



    def filter_vcf(args):
        from .variant_filtering import remove_overlapping_indels
        remove_overlapping_indels(args.vcf_file_name)


    subparser = subparsers.add_parser("remove_overlapping_indels")
    subparser.add_argument("-v", "--vcf-file-name", required=True)
    subparser.set_defaults(func=filter_vcf)




    subparser = subparsers.add_parser("create_helper_model")
    subparser.add_argument("-g", "--genotype-matrix", required=False)
    subparser.add_argument("-v", "--variant-to-nodes", required=True)
    subparser.add_argument("-m", "--most-similar-variants", required=False)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-t", "--n-threads", required=False, default=1, type=int)
    subparser.add_argument(
        "-u", "--use-duplicate-counts", required=False, type=bool, default=False
    )
    subparser.add_argument(
        "-w",
        "--window-size",
        required=False,
        default=50,
        type=int,
        help="Number of variants before/after considered as potential helper variant",
    )
    subparser.set_defaults(func=create_helper_model)

    def make_index_bundle(args):
        bundle = IndexBundle.from_args(args)
        bundle.to_file(args.out_file_name, compress=True)
        logging.info("Wrote index bundle to file %s" % args.out_file_name)

    subparser = subparsers.add_parser("make_index_bundle")
    subparser.add_argument("-g", "--variant-to-nodes", required=True)
    subparser.add_argument("-v", "--numpy-variants", required=True)
    subparser.add_argument("-A", "--count-model", required=True, help="Node count model")
    subparser.add_argument("-o", "--out-file-name", required=True, help="Will write genotyped variants to this file")
    subparser.add_argument("-x", "--tricky-variants", required=True)
    subparser.add_argument("-f", "--helper-model", required=True)
    subparser.add_argument("-F", "--helper-model-combo-matrix", required=True)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.set_defaults(func=make_index_bundle)

    subparser = subparsers.add_parser("sample_node_counts_from_population")
    subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-k", "--kmer-size", required=False, type=int, default=31)
    subparser.add_argument("-H", "--haplotype-to-nodes", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-t", "--n-threads", required=False, type=int, default=1)
    subparser.add_argument("-M", "--max-count", required=False, type=int, default=30)
    subparser.add_argument("-l", "--limit-to-n-individuals", required=False, type=int, default=0)
    subparser.set_defaults(func=sample_node_counts_from_population_cli)

    subparser = subparsers.add_parser("refine_sampling_model")
    subparser.add_argument("-s", "--sampling_model", required=True)
    subparser.add_argument("-v", "--variant-to-nodes", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=refine_sampling_model)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)
    remove_shared_memory_in_session()


