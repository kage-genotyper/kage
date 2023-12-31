import logging
import sys

from kage.genotyping.combination_model_genotyper import downscale_coverage

from .util import convert_string_genotypes_to_numeric_array, _write_genotype_debug_data

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
import argparse, time, random
from kage.models.helper_index import create_helper_model
from .configuration import GenotypingConfig
from kage.models.mapping_model import sample_node_counts_from_population_cli, refine_sampling_model, make_sparse_count_model
from shared_memory_wrapper import (
    get_shared_pool,
    close_shared_pool, )
from shared_memory_wrapper.shared_memory import remove_all_shared_memory
from graph_kmer_index import KmerIndex, ReverseKmerIndex
from kage.analysis.analysis import analyse_variants
import numpy as np
from .node_counts import NodeCounts
from obgraph.variant_to_nodes import VariantToNodes
from .indexing.tricky_variants import find_variants_with_nonunique_kmers, find_tricky_variants
from .indexing.index_bundle import IndexBundle
from .genotyping.combination_model_genotyper import CombinationModelGenotyper, add_svs_to_tricky_variants, set_uniform_probs_for_svs
from kmer_mapper.command_line_interface import map_bnp
from argparse import Namespace
from .indexing.sparse_haplotype_matrix import make_sparse_haplotype_matrix_cli
from kage.indexing.main import make_index_cli
from .analysis.debugging import debug_cli
from kage.io import create_vcf_header_with_sample_name, write_multiallelic_vcf_with_biallelic_numeric_genotypes
from kage.benchmarking.vcf_preprocessing import preprocess_sv_vcf, filter_snps_indels_covered_by_svs_cli
from kage.analysis.genotype_accuracy import genotype_accuracy_cli
from kage.benchmarking.vcf_preprocessing import filter_low_frequency_alleles_on_multiallelic_variants_cli
from kage.indexing.main import MultiAllelicMap
from kage.glimpse.glimpse_wrapper import run_glimpse_cli, run_glimpse_index_cli
from pathlib import Path
import os
from .glimpse.glimpse_wrapper import run_glimpse
from kage.naive_genotyper import naive_genotyper_cli
from kage.preprocessing.variants import convert_purebread_vcf

np.random.seed(1)
np.seterr(all="ignore")
np.set_printoptions(suppress=True)


def main():
    run_argument_parser(sys.argv[1:])


def get_kmer_counts(kmer_index, k, reads_file_name, n_threads, gpu=False):
    logging.info("Will count kmers.")
    # call kmer mapper
    return NodeCounts(map_bnp(Namespace(
        kmer_size=k, kmer_index=kmer_index, reads=reads_file_name, n_threads=n_threads, gpu=gpu, debug=False,
        chunk_size=10000000, map_reverse_complements=True if gpu else False, func=None, output_file=None
    )))


def genotype(args):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Will show debug")

    start_time = time.perf_counter()
    logging.info("Read coverage is set to %.3f" % args.average_coverage)
    get_shared_pool(args.n_threads)

    logging.info("Reading all indexes from an index bundle")
    t = time.perf_counter()
    index = IndexBundle.from_file(args.index_bundle).indexes

    if args.only_impute_svs:
        add_svs_to_tricky_variants(index)


    logging.debug("Reading indexes took %.3f sec" % (time.perf_counter()-t))
    config = GenotypingConfig.from_command_line_args(args)
    if args.glimpse is not None:
        config.ignore_helper_model = True
        config.ignore_helper_variants = True
        logging.info("Will do imputation with glimpse")
        assert args.glimpse.endswith(".vcf.gz"), "--glimpse parameter must point to a .vcf.gz file"
        assert os.path.isfile(args.glimpse + ".tbi"), "A tabix index file must exist for the glimpse vcf %s" % args.glimpse

    if not "helper_variants" in index:
        config.ignore_helper_model = True
        config.ignore_helper_variants = True
        logging.info("Did not find helper variants in index. Will not use helper variants/model")


    kmer_index = index.kmer_index
    assert args.reads is not None, "--reads must be specified if not node_counts is specified"
    node_counts = get_kmer_counts(kmer_index, args.kmer_size, args.reads, config.n_threads, args.gpu)


    np.save(args.out_file_name + ".node_counts.npy", node_counts.node_counts)

    if args.average_coverage > 3 and args.glimpse is not None:
        downscale_coverage(config, node_counts, 3)

    max_variant_id = len(index.variant_to_nodes.ref_nodes) - 1
    logging.info("Max variant id is assumed to be %d" % max_variant_id)

    genotyper = CombinationModelGenotyper(0, max_variant_id, node_counts, index, config=config)
    genotypes, probs, count_probs = genotyper.genotype()
    from kage.genotyping.multiallelic import postprocess_multiallelic_calls
    # Numeric genotypes: 1: 0/0, 2: 1/1, 3: 0/1
    #multiallelic_map = index.multiallelic_map
    multiallelic_map = MultiAllelicMap.from_variants_by_position(index.vcf_variants)
    genotypes, probs = postprocess_multiallelic_calls(genotypes, multiallelic_map, probs)

    t = time.perf_counter()
    numpy_genotypes = convert_string_genotypes_to_numeric_array(genotypes)
    logging.debug("Converting string genotypes to numeric took %.4f sec" % (time.perf_counter()-t))

    if args.write_debug_data:
        _write_genotype_debug_data(genotypes, numpy_genotypes, args.out_file_name, index.variant_to_nodes, probs, count_probs)

    # new setup: Storing SimpleVcfEntry object in index, use this to write vcf
    logging.info("Writing vcf using Vcf entry to %s" % args.out_file_name)
    out_file_name = args.out_file_name
    if args.glimpse is not None:
        out_file_name = os.path.splitext(out_file_name)[0] + "_no_imputation" + Path(args.out_file_name).suffix
        logging.info("Will use GLIMPSE. Writing original vcf to %s" % out_file_name)

    if args.only_impute_svs:
        # set all SV probs to uniform, meaning read information is ignored
        set_uniform_probs_for_svs(index.vcf_variants, probs)

    write_multiallelic_vcf_with_biallelic_numeric_genotypes(
        index.vcf_variants, genotypes, out_file_name,
        index.n_alleles_per_variant,
        header=create_vcf_header_with_sample_name(index.vcf_header, config.sample_name_output, add_genotype_likelyhoods=not config.do_not_write_genotype_likelihoods),
        add_genotype_likelihoods=probs if not config.do_not_write_genotype_likelihoods else None,
        ignore_homo_ref=config.ignore_homo_ref,
    )

    logging.info("Writing to vcf took %.3f sec" % (time.perf_counter() - t))

    if args.glimpse is not None:
        t0 = time.perf_counter()
        chromosomes = list(set([variant.chromosome.to_string() for variant in index.vcf_variants]))  # glimpse wrapper needs the unique chromosomes that we actually have variants for
        run_glimpse(args.glimpse, out_file_name, args.out_file_name, n_threads=args.n_threads,
                    chromosomes=chromosomes, glimpse_index_dir=args.glimpse_chunks)
        logging.info("Running GLIMPSE took %.4f sec" % (time.perf_counter()-t0))

    close_shared_pool()
    logging.info("Genotyping took %d sec" % (time.perf_counter() - start_time))


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
    subparser.add_argument("-r", "--reads", required=False)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=31)
    subparser.add_argument("-g", "--gpu", required=False, type=bool, default=False)
    subparser.add_argument("-i", "--index-bundle", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True, help="Will write genotyped variants to this file")
    subparser.add_argument("-t", "--n-threads", type=int, required=False, default=8)
    subparser.add_argument("-a", "--average-coverage", type=float, default=15, help="Expected average read coverage", )
    subparser.add_argument("-q", "--min-genotype-quality", type=float, default=0.0,
        help="Min prob of genotype being correct. Genotypes with prob less than this are set to homo ref.")
    subparser.add_argument( "-s", "--sample-name-output", required=False, default="DONOR", help="Sample name that will be used in the output vcf")
    subparser.add_argument( "-u", "--use-naive-priors", required=False, type=bool, default=False,
        help="Set to True to use only population allele frequencies as priors.")
    #subparser.add_argument("-l", "--limit-model-counts", default=0, type=int, help="If larger than 0, model will ignore counts larger than this. Can be used to use lower memory, but will make model less accurate.")
    subparser.add_argument("-I", "--ignore-helper-model", required=False, type=bool, default=False)
    subparser.add_argument("-V", "--ignore-helper-variants", required=False, type=bool, default=False)
    subparser.add_argument("-b", "--ignore-homo-ref", required=False, type=bool, default=False, help="Set to True to not write homo ref variants to output vcf")
    subparser.add_argument("-B", "--do-not-write-genotype-likelihoods", required=False, type=bool, default=False, help="Set to True to not write genotype likelihoods to output vcf")
    subparser.add_argument("-d", "--debug", type=bool, default=False)
    subparser.add_argument("-D", "--write-debug-data", type=bool, default=False)
    subparser.add_argument("-G", "--glimpse", default=None,
                           help="If set, GLIMPSE will be used as imputation instead of KAGE's builtin method.")
    subparser.add_argument("-c", "--glimpse-chunks", default=None,
                           help="Can be set to a directory created by running kage glimpse_index. If not set, this index will be created.")
    subparser.add_argument("-S", "--only-impute-svs", type=bool, default=False, help="If set to True, KAGE will ignore kmers for SVs and only base SV calsl on imputation. Meant to be run with --glimpse ...")

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


    def simulate_reads_cli(args):
        from kage.simulation.read_simulation import simulate_reads
        simulate_reads(args.vcf, args.fasta,
                       out_file_name=args.out_file_name,
                        coverage=args.coverage,
                        read_length=args.read_length,
                       snp_error_rate=args.snp_error_rate,
                       random_seed=args.random_seed,
                       paired_end=args.paired_end,
                       paired_end_insert_size=args.paired_end_insert_size,
                       paired_end_insert_sd=args.paired_end_insert_sd,
                       )

    subparser = subparsers.add_parser("simulate_reads")
    subparser.add_argument("-v", "--vcf", required=True)
    subparser.add_argument("-f", "--fasta", required=True)
    subparser.add_argument("-o", "--out_file_name", required=True)
    subparser.add_argument("-c", "--coverage", type=float, required=True)
    subparser.add_argument("-s", "--random-seed", type=int, required=False, default=1)
    subparser.add_argument("-e", "--snp-error-rate", type=float, required=False, default=0.001)
    subparser.add_argument("-l", "--read-length", type=int, required=False, default=150)
    subparser.add_argument("-p", "--paired-end", type=bool, required=False, default=False)
    subparser.add_argument("-i", "--paired-end-insert-size", type=int, required=False, default=500)
    subparser.add_argument("-d", "--paired-end-insert-sd", type=int, required=False, default=50)
    subparser.set_defaults(func=simulate_reads_cli)

    def preprocess_sv_vcf_cli(args):
        preprocess_sv_vcf(args.vcf, args.fasta)


    subparser = subparsers.add_parser("preprocess_sv_vcf")
    subparser.add_argument("-v", "--vcf", required=True)
    subparser.add_argument("-f", "--fasta", required=True)
    subparser.set_defaults(func=preprocess_sv_vcf_cli)


    subparser = subparsers.add_parser("filter_snps_indels_covered_by_svs")
    subparser.add_argument("-v", "--vcf", required=True)
    subparser.add_argument("-l", "--sv_size_limit", required=False, default=50, type=int)
    subparser.set_defaults(func=filter_snps_indels_covered_by_svs_cli)



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

    subparser = subparsers.add_parser("make_sparse_count_model")
    subparser.add_argument("-s", "--count_model", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=make_sparse_count_model)

    subparser = subparsers.add_parser("make_sparse_haplotype_matrix")
    subparser.add_argument("-v", "--vcf-file-name", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=make_sparse_haplotype_matrix_cli)

    subparser = subparsers.add_parser("index")
    subparser.add_argument("-r", "--reference", required=True)
    subparser.add_argument("-v", "--vcf", required=True)
    subparser.add_argument("-V", "--vcf-no-genotypes", required=False, help="May be specified. If specified, should be a vcf with no genotypes. Will lower memory usage when creating indexes. Will not change results.")
    subparser.add_argument("-o", "--out-base-name", required=True)
    subparser.add_argument("-k", "--kmer-size", required=False, type=int, default=31)
    subparser.add_argument("-m", "--modulo", required=False, type=int, default=200000033)
    subparser.add_argument("-w", "--variant-window", required=False, type=int, default=7, help="Max neighbouring variants to consider when selecting kmers. Indexing increases with this number. 6 or 7 should work fine for most cases.")
    subparser.add_argument("-a", "--min-af-deletions-filter", required=False, type=float, default=0.1, help="Deletions with lower allele frequency than this will not be indexed, to avoid too many duplicate alleles. Normally this parameter does not need to be changed.")
    subparser.add_argument("-H", "--no-helper-model", required=False, type=bool, default=False, help="Set to True to not create a helper model")
    subparser.add_argument("-t", "--n-threads", type=int, required=False, default=16)
    subparser.set_defaults(func=make_index_cli)


    subparser = subparsers.add_parser("debug")
    subparser.add_argument("-i", "--index", required=True)
    subparser.add_argument("-t", "--truth", required=True)
    subparser.add_argument("-g", "--genotypes", required=True)
    subparser.add_argument("-r", "--report", required=True)
    subparser.add_argument("-n", "--node-counts", required=True)
    subparser.add_argument("-p", "--probs", required=True)
    subparser.add_argument("-c", "--count-probs", required=True)
    subparser.add_argument("-G", "--numeric-genotypes", required=True)
    subparser.set_defaults(func=debug_cli)


    subparser = subparsers.add_parser("genotype_accuracy")
    subparser.add_argument("-t", "--truth", required=True)
    subparser.add_argument("-g", "--genotypes", required=True)
    subparser.add_argument("-v", "--limit-type-to", required=False, default="all")
    subparser.set_defaults(func=genotype_accuracy_cli)

    subparser = subparsers.add_parser("filter_low_freq_alleles")
    subparser.add_argument("-v", "--vcf-file-name", required=True)
    subparser.add_argument("-f", "--min-frequency", required=True, type=float)
    subparser.add_argument("-r", "--reference", required=False, default=None)
    subparser.add_argument("-d", "--only-deletions", required=False, default=False, type=bool)
    subparser.set_defaults(func=filter_low_frequency_alleles_on_multiallelic_variants_cli)

    subparser = subparsers.add_parser("glimpse", help="Wrapper around GLIMPSE 1")
    subparser.add_argument("-p", "--population-vcf", required=True)
    subparser.add_argument("-g", "--genotyped-vcf", required=True)
    #subparser.add_argument("-m", "--genetic-map-directory", required=False, default="")
    subparser.add_argument("-o", "--output-vcf", required=True, help="Will write genotypes to this vcf in bgzipped format")
    subparser.add_argument("-c", "--chromosomes", required=True, help="Comma-separated list of chromosomes")
    subparser.add_argument("-t", "--n-threads", type=int, required=False, default=8)
    subparser.set_defaults(func=run_glimpse_cli)

    subparser = subparsers.add_parser("glimpse_index", help="Makes chunks necessary for running GLIMPSE")
    subparser.add_argument("-p", "--population-vcf", required=True)
    subparser.add_argument("-o", "--output-dir", required=True,
                           help="Will write chunks to this directory")
    subparser.add_argument("-t", "--n-threads", type=int, required=False, default=8)
    subparser.set_defaults(func=run_glimpse_index_cli)

    subparser = subparsers.add_parser("naive_genotyper", help="A baseline genotyper that does nothing")
    subparser.add_argument("-p", "--population-vcf", required=True)
    subparser.add_argument("-o", "--output-vcf", required=True)
    subparser.add_argument("-g", "--glimpse", required=False, type=bool, default=False)
    subparser.add_argument("-c", "--glimpse-chunks", default=None,
                           help="Can be set to a directory created by running kage glimpse_index. If not set, this index will be created.")
    subparser.set_defaults(func=naive_genotyper_cli)

    subparser = subparsers.add_parser("convert_purebread_vcf")
    subparser.add_argument("-v", "--vcf-file-name", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=lambda args: convert_purebread_vcf(args.vcf_file_name, args.out_file_name))


    #subparser = subparsers.add_parser("pad_vcf")
    #subparser.add_argument("-v", "--vcf-file-name", required=True)
    #subparser.add_argument("-r", "--reference", required=True)
    #subparser.add_argument("-o", "--out-file-name", required=True)
    #subparser.set_defaults(func=pad_vcf_cli)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)
    #remove_shared_memory_in_session()


