import logging
import sys, argparse, time, itertools, math, random

from shared_memory_wrapper.util import interval_chunks

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

from kage.models.mapping_model import sample_node_counts_from_population_cli, refine_sampling_model
from kage.models.sampling_combo_model import LimitedFrequencySamplingComboModel

from shared_memory_wrapper.shared_memory import (
    from_file,
    from_shared_memory,
    to_shared_memory,
    remove_all_shared_memory,
    remove_shared_memory_in_session,
    get_shared_pool,
    close_shared_pool,
)
from graph_kmer_index import KmerIndex, ReverseKmerIndex
from kage.analysis.analysis import analyse_variants
from pathos.multiprocessing import Pool
import numpy as np

np.set_printoptions(suppress=True)
from .node_counts import NodeCounts
from .node_count_model import (
    NodeCountModel,
    GenotypeNodeCountModel,
)
from obgraph.genotype_matrix import (
    MostSimilarVariantLookup,
    GenotypeFrequencies,
)
from obgraph.variant_to_nodes import VariantToNodes
from kage.models.helper_index import (
    make_helper_model_from_genotype_matrix,
    make_helper_model_from_genotype_matrix_and_node_counts,
    HelperVariants,
    CombinationMatrix,
)
from obgraph.genotype_matrix import GenotypeMatrix
from obgraph.numpy_variants import NumpyVariants
from kage.indexing.tricky_variants import TrickyVariants, find_variants_with_nonunique_kmers, find_tricky_variants
from graph_kmer_index.index_bundle import IndexBundle
from .genotyping.combination_model_genotyper import CombinationModelGenotyper

np.random.seed(1)
np.seterr(all="ignore")


def main():
    run_argument_parser(sys.argv[1:])


def genotype(args):
    start_time = time.perf_counter()
    logging.info("Using genotyper %s" % args.genotyper)
    logging.info("Read coverage is set to %.3f" % args.average_coverage)
    get_shared_pool(args.n_threads)

    args.shared_memory_unique_id = str(random.randint(0, 1e15))
    logging.info("Random id for shared memory: %s" % args.shared_memory_unique_id)
    p = get_shared_pool(args.n_threads)  # Pool(16)
    genotype_frequencies = None

    models = None

    if args.index_bundle is not None:
        logging.info("Reading all indexes from an index bundle")
        index = IndexBundle.from_file(args.index_bundle, skip=["KmerIndex"]).indexes
        models = index["CountModel"]
        variant_to_nodes = index["VariantToNodes"]
        variants = index["NumpyVariants"]
        helper_model = index["HelperVariants"].helper_variants
        helper_model_combo_matrix = index["CombinationMatrix"].matrix
        tricky_variants = index["TrickyVariants"].tricky_variants
    else:
        logging.info("Not reading indexes from bundle, but from separate files")

        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)

        if args.count_model is None:
            logging.info("Model not specified. Creating a naive model, assuming kmers are unique")
            args.count_model = [
                LimitedFrequencySamplingComboModel.create_naive(len(variant_to_nodes.ref_nodes)),
                LimitedFrequencySamplingComboModel.create_naive(len(variant_to_nodes.var_nodes))
            ]
        else:
            args.count_model = from_file(args.count_model)

        if args.count_model is not None:
            models = args.count_model

        variants = NumpyVariants.from_file(args.vcf)

        tricky_variants = None
        if args.tricky_variants is not None:
            logging.info("Using tricky variants")
            tricky_variants = TrickyVariants.from_file(
                args.tricky_variants
            ).tricky_variants

        helper_model = None
        helper_model_combo_matrix = None
        if args.helper_model is not None:
            helper_model = HelperVariants.from_file(args.helper_model).helper_variants
            helper_model_combo_matrix = np.load(args.helper_model_combo_matrix)

        if args.genotype_frequencies is not None:
            genotype_frequencies = GenotypeFrequencies.from_file(
                args.genotype_frequencies
            )

    assert models is not None

    node_counts = NodeCounts.from_file(args.counts)
    max_variant_id = len(variant_to_nodes.ref_nodes) - 1
    logging.info("Max variant id is assumed to be %d" % max_variant_id)
    # variant_chunks = list([int(i) for i in np.linspace(0, max_variant_id, args.n_threads + 1)])
    # variant_chunks = [(from_pos, to_pos) for from_pos, to_pos in zip(variant_chunks[0:-1], variant_chunks[1:])]
    variant_chunks = [(0, max_variant_id)]
    logging.info("Will genotype intervals %s" % variant_chunks)

    genotyper_class = CombinationModelGenotyper if args.genotyper is not None else globals()[args.genotyper]

    genotyper = genotyper_class(
        models,  # should be one model for ref node and one for var node in a list
        0,
        max_variant_id,
        variant_to_nodes,
        node_counts,
        genotype_frequencies,
        None,
        avg_coverage=args.average_coverage,
        genotype_transition_probs=None,
        tricky_variants=tricky_variants,
        use_naive_priors=args.use_naive_priors,
        helper_model=helper_model,
        helper_model_combo=helper_model_combo_matrix,
        n_threads=args.n_threads,
        ignore_helper_model=args.ignore_helper_model,
        ignore_helper_variants=args.ignore_helper_variants,
    )
    genotypes, probs, count_probs = genotyper.genotype()

    if args.min_genotype_quality > 0.0:
        set_to_homo_ref = math.e ** np.max(probs, axis=1) < args.min_genotype_quality
        logging.warning(
            "%d genotypes have lower prob than %.4f. Setting these to homo ref."
            % (np.sum(set_to_homo_ref), args.min_genotype_quality)
        )
        genotypes[set_to_homo_ref] = 0


    numeric_genotypes = ["0/0", "0/0", "1/1", "0/1"]
    numpy_genotypes = np.array([numeric_genotypes[g] for g in genotypes], dtype="|S3")
    variants.to_vcf_with_genotypes(
        args.out_file_name,
        args.sample_name_output,
        numpy_genotypes,
        add_header_lines=['##FILTER=<ID=LowQUAL,Description="Quality is low">',
                          '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="PHRED-scaled genotype likelihoods.">',
                          '##FORMAT=<ID=GL,Number=G,Type=Float,Description="Genotype likelihoods.">'
                          ],
        ignore_homo_ref=args.ignore_homo_ref,
        add_genotype_likelyhoods=probs if not args.do_not_write_genotype_likelihoods else None,
    )

    close_shared_pool()
    logging.info("Genotyping took %d sec" % (time.perf_counter() - start_time))
    np.save(args.out_file_name + ".probs", probs)
    np.save(args.out_file_name + ".count_probs", count_probs)

    # Make arrays with haplotypes
    haplotype_array1 = np.zeros(len(numpy_genotypes), dtype=np.uint8)
    haplotype_array1[np.where((genotypes == 2) | (genotypes == 3))[0]] = 1
    haplotype_array2 = np.zeros(len(numpy_genotypes), dtype=np.uint8)
    haplotype_array2[np.where(genotypes == 2)[0]] = 1
    np.save(args.out_file_name + ".haplotype1", haplotype_array1)
    np.save(args.out_file_name + ".haplotype2", haplotype_array2)

    # also store variant nodes from the two haplotypes
    variant_nodes_haplotype1 = variant_to_nodes.var_nodes[np.nonzero(haplotype_array1)]
    variant_nodes_haplotype2 = variant_to_nodes.var_nodes[np.nonzero(haplotype_array2)]
    np.save(args.out_file_name + ".haplotype1_nodes", variant_nodes_haplotype1)
    np.save(args.out_file_name + ".haplotype2_nodes", variant_nodes_haplotype2)


def run_argument_parser(args):
    parser = argparse.ArgumentParser(
        description="Alignment free graph genotyper",
        prog="alignment_free_graph_genotyper",
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=50, width=100
        ),
    )

    subparsers = parser.add_subparsers()

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

    subparser = subparsers.add_parser("genotype")
    subparser.add_argument("-c", "--counts", required=True)
    subparser.add_argument(
        "-i",
        "--index-bundle",
        required=False,
        help="If set, needs to be a bundle of all the indexes. If not set, other indexes needs to be specified.",
    )
    subparser.add_argument("-g", "--variant-to-nodes", required=False)
    subparser.add_argument("-v", "--vcf", required=False, help="Vcf to genotype")
    subparser.add_argument("-m", "--model", required=False, help="Node count model")
    subparser.add_argument(
        "-A", "--count-model", required=False, help="Node count model"
    )
    subparser.add_argument(
        "-G", "--genotype-frequencies", required=False, help="Genotype frequencies"
    )

    subparser.add_argument(
        "-o",
        "--out-file-name",
        required=True,
        help="Will write genotyped variants to this file",
    )
    subparser.add_argument(
        "-C",
        "--genotyper",
        required=False,
        default="CombinationModelGenotyper",
        help="Genotyper to use",
    )
    subparser.add_argument("-t", "--n-threads", type=int, required=False, default=8)
    subparser.add_argument(
        "-a",
        "--average-coverage",
        type=float,
        default=15,
        help="Expected average read coverage",
    )
    subparser.add_argument(
        "-q",
        "--min-genotype-quality",
        type=float,
        default=0.0,
        help="Min prob of genotype being correct. Genotypes with prob less than this are set to homo ref.",
    )
    subparser.add_argument("-p", "--genotype-transition-probs", required=False)
    subparser.add_argument("-x", "--tricky-variants", required=False)
    subparser.add_argument(
        "-s",
        "--sample-name-output",
        required=False,
        default="DONOR",
        help="Sample name that will be used in the output vcf",
    )
    subparser.add_argument(
        "-u",
        "--use-naive-priors",
        required=False,
        type=bool,
        default=False,
        help="Set to True to use only population allele frequencies as priors.",
    )
    subparser.add_argument("-f", "--helper-model", required=False)
    subparser.add_argument("-F", "--helper-model-combo-matrix", required=False)
    subparser.add_argument("-I", "--ignore-helper-model", required=False, type=bool, default=False)
    subparser.add_argument("-V", "--ignore-helper-variants", required=False, type=bool, default=False)
    subparser.add_argument("-b", "--ignore-homo-ref", required=False, type=bool, default=False, help="Set to True to not write homo ref variants to output vcf")
    subparser.add_argument("-B", "--do-not-write-genotype-likelihoods", required=False, type=bool, default=False, help="Set to True to not write genotype likelihoods to output vcf")

    subparser.set_defaults(func=genotype)

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

    def make_genotype_model(args):
        node_counts = NodeCountModel.from_file(args.node_count_model)
        variant_nodes = VariantToNodes.from_file(args.variant_to_nodes)
        genotype_model = GenotypeNodeCountModel.from_node_count_model(
            node_counts, variant_nodes
        )
        genotype_model.to_file(args.out_file_name)

    subparser = subparsers.add_parser("make_genotype_model")
    subparser.add_argument("-n", "--node-count-model", required=True)
    subparser.add_argument("-v", "--variant-to-nodes", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=make_genotype_model)


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


    def create_helper_model_single_thread(data):
        interval, args = data
        from_variant, to_variant = interval

        variant_to_nodes = from_shared_memory(
            VariantToNodes, "variant_to_nodes" + args.shared_memory_unique_id
        )
        genotype_matrix = from_shared_memory(
            GenotypeMatrix, "genotype_matrix" + args.shared_memory_unique_id
        )

        # read genotype matrix etc from shared memory
        # submatrix = GenotypeMatrix(genotype_matrix.matrix[from_variant:to_variant,:])
        submatrix = GenotypeMatrix(
            genotype_matrix.matrix[
                from_variant:to_variant:,
            ]
        )
        logging.info(
            "Creating helper model for %d individuals and %d variants"
            % (submatrix.matrix.shape[1], submatrix.matrix.shape[0])
        )
        sub_variant_to_nodes = variant_to_nodes.slice(from_variant, to_variant)
        use_duplicate_counts = args.use_duplicate_counts

        subhelpers, subcombo = make_helper_model_from_genotype_matrix(
            submatrix.matrix, None, dummy_count=1.0, window_size=args.window_size
        )

        # variant ids in results are now from 0 to (to_variant-from_variant)
        subhelpers += from_variant
        return from_variant, to_variant, subhelpers, subcombo


    def create_helper_model(args):
        args.shared_memory_unique_id = str(random.randint(0, 1e15))
        pool = Pool(args.n_threads)
        logging.info("Made pool")
        model = None

        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
        genotype_matrix = GenotypeMatrix.from_file(args.genotype_matrix)
        # NB: Transpose
        genotype_matrix.matrix = genotype_matrix.matrix.transpose()

        if args.n_threads > 1:
            n_variants = len(variant_to_nodes.ref_nodes)
            n_threads = args.n_threads
            while n_variants < n_threads * 50 and n_threads > 2:
                n_threads -= 1
                logging.info("Lowered n threads to %d so that not too few variants are analysed together" % n_threads)

            variant_intervals = interval_chunks(0, n_variants, n_threads)
            logging.info("Will process variant intervals: %s" % variant_intervals)

            helpers = np.zeros(n_variants, dtype=np.uint32)
            genotype_matrix_combo = np.zeros((n_variants, 3, 3), dtype=float)

            logging.info("Putting data in shared memory")
            # put data in shared memory
            to_shared_memory(
                genotype_matrix, "genotype_matrix" + args.shared_memory_unique_id
            )
            to_shared_memory(
                variant_to_nodes, "variant_to_nodes" + args.shared_memory_unique_id
            )

            logging.info("Put data in shared memory")

            for from_variant, to_variant, subhelpers, subcombo in pool.imap(
                create_helper_model_single_thread, zip(variant_intervals, itertools.repeat(args))
            ):
                logging.info("Done with one chunk")
                helpers[from_variant:to_variant] = subhelpers
                genotype_matrix_combo[from_variant:to_variant] = subcombo

        else:
            (
                helpers,
                genotype_matrix_combo,
            ) = make_helper_model_from_genotype_matrix_and_node_counts(
                genotype_matrix, model, variant_to_nodes, args.window_size
            )
        genotype_matrix_combo = genotype_matrix_combo.astype(np.float32)

        np.save(args.out_file_name, helpers)
        logging.info("Saved helper model to file: %s" % args.out_file_name)
        np.save(args.out_file_name + "_combo_matrix", genotype_matrix_combo)
        logging.info(
            "Saved combo matrix to file %s" % args.out_file_name + "_combo_matrix"
        )

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
        indexes = {
            "VariantToNodes": VariantToNodes.from_file(args.variant_to_nodes),
            "NumpyVariants": NumpyVariants.from_file(args.numpy_variants),
            "CountModel": from_file(
                args.count_model
            ),
            "TrickyVariants": TrickyVariants.from_file(args.tricky_variants),
            "HelperVariants": HelperVariants.from_file(args.helper_model),
            "CombinationMatrix": CombinationMatrix.from_file(
                args.helper_model_combo_matrix
            ),
            "KmerIndex": KmerIndex.from_file(args.kmer_index),
        }
        bundle = IndexBundle(indexes)
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


