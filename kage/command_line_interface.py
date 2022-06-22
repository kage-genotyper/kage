import logging
import sys
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)

from .util import log_memory_usage_now
from .sampling_combo_model import RaggedFrequencySamplingComboModel
from .models import ComboModelBothAlleles
from .mapping_model import get_node_counts_from_haplotypes
from .mapping_model import get_sampled_nodes_and_counts
from .sampling_combo_model import LimitedFrequencySamplingComboModel


import itertools
from itertools import repeat
import sys, argparse, time
from shared_memory_wrapper.shared_memory import (
    from_shared_memory,
    to_shared_memory,
    remove_shared_memory,
    SingleSharedArray,
    remove_all_shared_memory,
    remove_shared_memory_in_session,
    get_shared_pool,
    close_shared_pool,
)
from obgraph import Graph as ObGraph
from graph_kmer_index import KmerIndex, ReverseKmerIndex
from graph_kmer_index import ReferenceKmerIndex
from .analysis import GenotypeDebugger
from obgraph.variants import VcfVariants, TruthRegions
from obgraph.haplotype_nodes import HaplotypeToNodes, DiscBackedHaplotypeToNodes
from .reads import read_chunks_from_fasta
import platform
from pathos.multiprocessing import Pool
import numpy as np

np.set_printoptions(suppress=True)
from .node_counts import NodeCounts
from .node_count_model import (
    NodeCountModel,
    GenotypeNodeCountModel,
    NodeCountModelAlleleFrequencies,
    NodeCountModelAdvanced,
    NodeCountModelCreatorAdvanced,
)
from obgraph.genotype_matrix import (
    MostSimilarVariantLookup,
    GenotypeFrequencies,
    GenotypeTransitionProbabilities,
)
from obgraph.variant_to_nodes import VariantToNodes
from .genotyper import Genotyper
from .numpy_genotyper import NumpyGenotyper
from .combination_model_genotyper import CombinationModelGenotyper
import SharedArray as sa
from obgraph.haplotype_matrix import HaplotypeMatrix
from obgraph.variant_to_nodes import NodeToVariants
import random
from obgraph.genotype_matrix import GenotypeMatrix
from .helper_index import (
    make_helper_model_from_genotype_matrix,
    make_helper_model_from_genotype_matrix_and_node_counts,
    HelperVariants,
    CombinationMatrix,
)
from obgraph.genotype_matrix import GenotypeMatrix
from obgraph.numpy_variants import NumpyVariants
from .tricky_variants import TrickyVariants
from graph_kmer_index.index_bundle import IndexBundle
from shared_memory_wrapper import from_file, to_file
import math
from .gaf_parsing import node_counts_from_gaf, parse_gaf
import pickle

np.random.seed(1)
np.seterr(all="ignore")


def main():
    run_argument_parser(sys.argv[1:])


def analyse_variants(args):
    from .node_count_model import NodeCountModel
    from obgraph.genotype_matrix import MostSimilarVariantLookup
    from obgraph.variant_to_nodes import VariantToNodes
    from .helper_index import CombinationMatrix

    whitelist = None
    # pangenie = VcfVariants.from_vcf(args.pangenie)

    logging.info("Reading variant nodes")
    variant_nodes = VariantToNodes.from_file(args.variant_nodes)
    logging.info("Reading kmer index")
    kmer_index = KmerIndex.from_file(args.kmer_index)
    logging.info("Reading reverse index")
    reverse_index = ReverseKmerIndex.from_file(args.reverse_index)
    logging.info("Reading model")
    #model = NodeCountModelAdvanced.from_file(args.model)
    model = from_file(args.model)
    logging.info(type(model))
    #model.astype(float)
    #model.fill_empty_data()

    logging.info("REading helper variants")
    helper_variants = np.load(args.helper_variants)
    logging.info("Reading combination matrix")
    combination_matrix = CombinationMatrix.from_file(args.combination_matrix)
    logging.info("Reading probs")
    probs = np.load(args.probs)
    logging.info("Reading count probs")
    count_probs = np.load(args.count_probs)

    logging.info("REading predicted genotyppes")
    predicted_genotypes = VcfVariants.from_vcf(args.predicted_vcf)

    logging.info("Reading true genotypes")
    true_genotypes = VcfVariants.from_vcf(args.truth_vcf)

    logging.info("Reading all genotypes")
    all_variants = VcfVariants.from_vcf(args.vcf)

    analyser = GenotypeDebugger(
        variant_nodes,
        args.kmer_size,
        all_variants,
        kmer_index,
        reverse_index,
        predicted_genotypes,
        true_genotypes,
        TruthRegions(args.truth_regions_file),
        NodeCounts.from_file(args.node_counts),
        model,
        helper_variants,
        combination_matrix,
        probs,
        count_probs,
        None,
    )
    analyser.analyse_unique_kmers_on_variants()


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

        if args.most_similar_variant_lookup is not None:
            most_similar_variant_lookup = MostSimilarVariantLookup.from_file(
                args.most_similar_variant_lookup
            )

        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)

        if args.count_model is None:
            logging.info("Model not specified. Creating a naive model, assuming kmers are unique")
            args.count_model = [
                LimitedFrequencySamplingComboModel.create_naive(len(variant_to_nodes.ref_nodes)),
                LimitedFrequencySamplingComboModel.create_naive(len(variant_to_nodes.var_nodes))
            ]
        else:
            t0 = time.perf_counter()
            args.count_model = from_file(args.count_model)
            logging.info("Took %.4f sec to read model" % (time.perf_counter()-t0))

        if args.count_model is not None:
            models = args.count_model



        t0 = time.perf_counter()
        variants = NumpyVariants.from_file(args.vcf)
        logging.info("Took %.3f sec to read variants" % (time.perf_counter()-t0))

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

    results = []
    genotyper_class = globals()[args.genotyper]
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
        set_to_homo_ref = np.where(
            math.e ** np.max(probs, axis=1) < args.min_genotype_quality
        )[0]
        logging.warning(
            "%d genotypes have lower prob than %.4f. Setting these to homo ref."
            % (len(set_to_homo_ref), args.min_genotype_quality)
        )
        logging.warning(
            "N non homo ref genotypes before: %d" % len(np.where(genotypes > 0)[0])
        )
        genotypes[set_to_homo_ref] = 0
        logging.warning(
            "N non homo ref genotypes after: %d" % len(np.where(genotypes > 0)[0])
        )

    # numpy_genotypes = np.empty(max_variant_id+1, dtype="|S3")
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


def model_using_kmer_index(variant_id_interval, args):
    variant_start_id, variant_end_id = variant_id_interval
    logging.info(
        "Processing variants with id between %d and %d"
        % (variant_start_id, variant_end_id)
    )
    from .node_count_model import NodeCountModel

    allele_frequency_index = None
    if args.allele_frequency_index is not None:
        allele_frequency_index = np.load(args.allele_frequency_index)

    haplotype_matrix = None
    node_to_variants = None
    if args.haplotype_matrix is not None:
        haplotype_matrix = HaplotypeMatrix.from_file(args.haplotype_matrix)
        node_to_variants = NodeToVariants.from_file(args.node_to_variants)

    if args.version == "":
        model_class = NodeCountModelCreatorFromNoChaining
    elif args.version == "v3":
        model_class = NodeCountModelCreatorAdvanced
    elif args.version == "v2":
        model_class = NodeCountModelCreatorFromNoChainingOnlyAlleleFrequencies
        logging.warning(
            "Using new version which gets sum of allele frequencies and squared sum"
        )

    model_creator = model_class(
        from_shared_memory(KmerIndex, "kmer_index_shared"),
        from_shared_memory(ReverseKmerIndex, "reverse_index_shared"),
        from_shared_memory(VariantToNodes, "variant_to_nodes_shared"),
        variant_start_id,
        variant_end_id,
        args.max_node_id,
        scale_by_frequency=args.scale_by_frequency,
        allele_frequency_index=allele_frequency_index,
        haplotype_matrix=haplotype_matrix,
        node_to_variants=node_to_variants,
    )
    model_creator.create_model()
    return model_creator.get_results()


def model_using_kmer_index_multiprocess(args):
    reverse_index = ReverseKmerIndex.from_file(args.reverse_node_kmer_index)
    to_shared_memory(reverse_index, "reverse_index_shared")
    index = KmerIndex.from_file(args.kmer_index)
    to_shared_memory(index, "kmer_index_shared")
    variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
    to_shared_memory(variant_to_nodes, "variant_to_nodes_shared")

    max_node_id = args.max_node_id

    logging.info("Will use %d threads" % args.n_threads)
    # variants = VcfVariants.from_vcf(args.vcf, skip_index=True, make_generator=True)
    # variants = variants.get_chunks(chunk_size=args.chunk_size)

    n_threads = args.n_threads
    n_variants = len(variant_to_nodes.ref_nodes)
    intervals = [int(i) for i in np.linspace(0, n_variants, n_threads)]
    variant_intervals = [
        (from_id, to_id) for from_id, to_id in zip(intervals[0:-1], intervals[1:])
    ]
    logging.info("Will process variant intervals: %s" % variant_intervals)
    data_to_process = zip(variant_intervals, repeat(args))

    if args.version == "":
        expected_node_counts_not_following_node = np.zeros(max_node_id + 1, dtype=float)
        expected_node_counts_following_node = np.zeros(max_node_id + 1, dtype=float)
    elif args.version == "v3":
        resulting_model = NodeCountModelAdvanced.create_empty(max_node_id)
    elif args.version == "v2":
        allele_frequencies = np.zeros(max_node_id + 1, dtype=float)
        allele_frequencies_squared = np.zeros(max_node_id + 1, dtype=float)

    pool = Pool(args.n_threads)

    while True:
        results = pool.starmap(
            model_using_kmer_index, itertools.islice(data_to_process, args.n_threads)
        )
        if results:
            for res in results:
                if args.version == "":
                    expected_node_counts_following_node += res[0]
                    expected_node_counts_not_following_node += res[1]
                elif args.version == "v3":
                    resulting_model = resulting_model + res
                elif args.version == "v2":
                    allele_frequencies += res[0]
                    allele_frequencies_squared += res[1]
        else:
            logging.info("No results, breaking")
            break

    if args.version == "":
        model = NodeCountModel(
            expected_node_counts_following_node, expected_node_counts_not_following_node
        )
    elif args.version == "v3":
        model = resulting_model
    elif args.version == "v2":
        model = NodeCountModelAlleleFrequencies(
            allele_frequencies, allele_frequencies_squared
        )

    model.to_file(args.out_file_name)
    logging.info("Wrote model to %s" % args.out_file_name)


def model_for_read_mapping(args):
    variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
    ref_nodes = variant_to_nodes.ref_nodes
    var_nodes = variant_to_nodes.var_nodes
    max_node_id = max([np.max(ref_nodes), np.max(var_nodes)])

    frequencies = np.zeros(max_node_id + 1, dtype=float)
    frequencies_squared = np.zeros(max_node_id + 1, dtype=float)
    certain = np.zeros(max_node_id + 1, dtype=float)
    frequency_matrix = np.zeros((max_node_id + 1, 5), dtype=float)
    has_too_many = np.zeros(max_node_id + 1, dtype=bool)

    # simple naive model: No duplicate counts, all nodes have only 1 certain
    model = NodeCountModelAdvanced(
        frequencies, frequencies_squared, certain, frequency_matrix, has_too_many
    )
    model.to_file(args.out_file_name)
    logging.info("Wrote model to %s" % args.out_file_name)


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
    # subparser.add_argument("-a", "--pangenie", required=False)
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
        "-M",
        "--most_similar_variant_lookup",
        required=False,
        help="Most similar variant lookup",
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
        from .simulation import run_genotyper_on_simualated_data

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

    def find_tricky_variants(args):
        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
        # model = GenotypeNodeCountModel.from_file(args.node_count_model)
        #model = NodeCountModelAdvanced.from_file(args.node_count_model)
        model = from_file(args.node_count_model)
        reverse_index = ReverseKmerIndex.from_file(args.reverse_kmer_index)

        tricky_variants = np.zeros(len(variant_to_nodes.ref_nodes + 1), dtype=np.uint32)

        n_tricky_model = 0
        n_tricky_kmers = 0
        n_nonunique = 0

        max_counts_model = args.max_counts_model

        for variant_id in range(0, len(variant_to_nodes.ref_nodes)):
            if variant_id % 100000 == 0:
                logging.info(
                    "%d variants processed, %d tricky due to model, %d tricky due to kmers. N non-unique filtered: %d"
                    % (variant_id, n_tricky_model, n_tricky_kmers, n_nonunique)
                )

            ref_node = variant_to_nodes.ref_nodes[variant_id]
            var_node = variant_to_nodes.var_nodes[variant_id]

            #model_counts_ref = 1 + model.certain[ref_node] + model.frequencies[ref_node]
            #model_counts_var = 1 + model.certain[var_node] + model.frequencies[var_node]

            if args.only_allow_unique:
                # if model.counts_homo_ref[var_node] > 0 or model.counts_homo_alt[ref_node] > 0:
                if model.has_duplicates(ref_node) or model.has_duplicates(var_node):
                #if model_counts_ref > 1 or model_counts_var > 1:
                    n_nonunique += 1
                    tricky_variants[variant_id] = 1

            # if model_counts_ref[2] > max_counts_model and model_counts_var[2] > max_counts_model:
            # if model_counts_ref[2] < model_counts_ref[1] * 1.1 or model_counts_var[2] < model_counts_var[1] * 1.1:
            m = args.max_counts_model
            if model.has_no_data(ref_node) or model.has_no_data(var_node):
                # logging.warning(model_counts_ref)
                # logging.warning(model_counts_ref)
                tricky_variants[variant_id] = 1
                #print(model[1][ref_node], model[1][var_node])
                n_tricky_model += 1
            else:
                reference_kmers = set(reverse_index.get_node_kmers(ref_node))
                variant_kmers = set(reverse_index.get_node_kmers(var_node))
                if len(reference_kmers.intersection(variant_kmers)) > 0:
                    # logging.warning("-----\nKmer crash on variant %d \n Ref kmers: %s\n Var kmers: %s" % (variant_id, reference_kmers, variant_kmers))
                    tricky_variants[variant_id] = 1
                    n_tricky_kmers += 1


        logging.info(
            "Stats: %d tricky due to model, %d tricky due to kmers. N non-unique filtered: %d"
            % (n_tricky_model, n_tricky_kmers, n_nonunique)
        )

        TrickyVariants(tricky_variants).to_file(args.out_file_name)
        # np.save(args.out_file_name, tricky_variants)
        logging.info("Wrote tricky variants to file %s" % args.out_file_name)

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


    def find_variants_with_nonunique_kmers(args):
        output = np.zeros(len(args.variant_to_nodes.ref_nodes), dtype=np.uint8)
        n_filtered = 0
        n_sharing_kmers = 0

        for i, (ref_node, var_node) in enumerate(zip(args.variant_to_nodes.ref_nodes, args.variant_to_nodes.var_nodes)):
            reference_kmers = args.reverse_kmer_index.get_node_kmers(ref_node)
            variant_kmers = args.reverse_kmer_index.get_node_kmers(var_node)

            if i % 1000 == 0:
                logging.info("%d variants processed, %d filtered, %d sharing kmers" % (i, n_filtered, n_sharing_kmers))

            frequencies_ref = np.array([args.population_kmer_index.get_frequency(k)-1 for k in reference_kmers])
            frequencies_var = np.array([args.population_kmer_index.get_frequency(k)-1 for k in variant_kmers])
            #print(frequencies_ref, frequencies_var)
            #if sum(frequencies_ref) > 0 or sum(frequencies_var) > 0:
            if np.all(frequencies_ref > 0) or np.all(frequencies_var > 0):
                n_filtered += 1
                output[i] = 1
            elif len(set(reference_kmers).intersection(variant_kmers)) > 0:
                n_sharing_kmers += 1
                output[i] = 1

        TrickyVariants(output).to_file(args.out_file_name)
        #np.save(args.out_file_name, output)
        logging.info("Saved array with variants to %s" % args.out_file_name)


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

    def analyse_kmer_index(args):

        reverse_kmers = ReverseKmerIndex.from_file(args.reverse_kmer_index)
        index = KmerIndex.from_file(args.kmer_index)
        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
        from .variant_kmer_analyser import VariantKmerAnalyser

        analyser = VariantKmerAnalyser(
            reverse_kmers, index, variant_to_nodes, args.write_good_variants_to_file
        )
        analyser.analyse()
        logging.info("Done")

    # Analyse variant kmers
    subparser = subparsers.add_parser("analyse_kmer_index")
    subparser.add_argument("-r", "--reverse-kmer-index", required=True)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-g", "--variant-to-nodes", required=True)
    subparser.add_argument(
        "-o",
        "--write-good-variants-to-file",
        required=False,
        help="When specified, good variant IDs will be written to file",
    )
    subparser.set_defaults(func=analyse_kmer_index)

    subparser = subparsers.add_parser("model")
    subparser.add_argument("-g", "--variant-to-nodes", required=True)
    subparser.add_argument("-N", "--node-to-variants", required=False)
    subparser.add_argument("-H", "--haplotype-matrix", required=False)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-m", "--max-node-id", type=int, required=True)
    subparser.add_argument("-r", "--reverse_node_kmer_index", required=True)
    # subparser.add_argument("-v", "--vcf", required=True)
    # subparser.add_argument("-c", "--chunk-size", type=int, default=100000, help="Number of variants to process in each chunk")
    subparser.add_argument("-t", "--n-threads", type=int, default=1, required=False)
    subparser.add_argument(
        "-f", "--scale-by-frequency", required=False, type=bool, default=False
    )
    subparser.add_argument("-a", "--allele-frequency-index", required=False)
    subparser.add_argument("-V", "--version", required=False, default="v3")
    subparser.set_defaults(func=model_using_kmer_index_multiprocess)

    subparser = subparsers.add_parser("model_for_read_mapping")
    subparser.add_argument("-g", "--variant-to-nodes", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=model_for_read_mapping)

    def model_for_read_mapping_using_sampled_reads(args):
        from numpy_alignments import NumpyAlignments

        graph = args.graph
        edge_mapping = pickle.load(open(args.edge_mapping, "rb"))
        mappings = parse_gaf(args.gaf, edge_mapping)
        true_read_positions = NumpyAlignments.from_file(args.true_read_positions)

        n_skipped_low_score = 0
        n_mapped_back_to_origin = 0
        node_counts = np.zeros(len(graph.nodes) + 1, dtype=int)

        for i, mapping in enumerate(parse_gaf(args.gaf)):
            if i % 1000 == 0:
                logging.info("%d mappings processed" % i)

            if mapping.score < args.min_score:
                n_skipped_low_score += 1
                continue

            read_id = int(mapping.read_id)

            # check if mapping is at ca same location as sampled read
            sampled_read_chr = true_read_positions.chromosomes[read_id]
            sampled_read_start = true_read_positions.positions[read_id]
            sampled_read_end = sampled_read_start + mapping.read_length
            nodes_from_sampled_area = graph.get_linear_ref_nodes_between_offsets(
                sampled_read_chr, sampled_read_start, sampled_read_end
            )

            # print("Mapping nodes: %s, nodes from area: %s" % (mapping.nodes, nodes_from_sampled_area))

            if len(set(nodes_from_sampled_area).intersection(mapping.nodes)) > 0:
                # is from same area
                n_mapped_back_to_origin += 1
                continue

            node_counts[mapping.nodes] += 1

        model = NodeCountModelAdvanced.create_empty(graph.max_node_id())
        model.certain = np.round(node_counts / args.coverage)
        model.to_file(args.out_file_name)
        logging.info(
            "N reads skipped because low score: %d ( < %d)"
            % (n_skipped_low_score, args.min_score)
        )
        logging.info(
            "N reads that mapped back to origin (should be approx. equal to n reads mapped): %d"
            % n_mapped_back_to_origin
        )
        logging.info("Saved model to %s" % args.out_file_name)

    subparser = subparsers.add_parser("model_for_read_mapping_using_sampled_reads")
    subparser.add_argument("-g", "--graph", required=True, type=ObGraph.from_file)
    subparser.add_argument("-r", "--gaf", required=True)
    subparser.add_argument("-p", "--true-read-positions", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-e", "--edge-mapping", required=True)
    subparser.add_argument("-m", "--min-score", required=True, type=int)
    subparser.add_argument(
        "-c",
        "--coverage",
        required=True,
        type=int,
        help="Average coverage of sampled reads",
    )
    subparser.set_defaults(func=model_for_read_mapping_using_sampled_reads)

    def model_using_transition_probs(args):
        from .node_count_model import GenotypeModelCreatorFromTransitionProbabilities
        from obgraph.variant_to_nodes import NodeToVariants

        graph = ObGraph.from_file(args.graph)
        genotype_matrix = GenotypeMatrix.from_file(args.genotype_matrix)
        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
        node_to_variants = NodeToVariants.from_file(args.node_to_variants)
        mapping_index = KmerIndex.from_file(args.mapping_index)
        population_kemrs = KmerIndex.from_file(args.population_kmers)

        maker = GenotypeModelCreatorFromTransitionProbabilities(
            graph,
            genotype_matrix,
            variant_to_nodes,
            node_to_variants,
            mapping_index,
            population_kemrs,
            args.max_node_id,
        )

        maker.get_node_counts()
        genotype_model = GenotypeNodeCountModel(
            maker.counts_homo_ref, maker.counts_homo_alt, maker.counts_hetero
        )
        genotype_model.to_file(args.out_file_name)

    subparser = subparsers.add_parser("model_using_kmer_index2")
    subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-G", "--genotype_matrix", required=True)
    subparser.add_argument("-v", "--variant-to-nodes", required=True)
    subparser.add_argument("-V", "--node_to_variants", required=True)
    subparser.add_argument("-i", "--mapping-index", required=True)
    subparser.add_argument("-I", "--population-kmers", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-m", "--max-node-id", type=int, required=True)
    subparser.add_argument("-t", "--n-threads", type=int, default=1, required=False)
    subparser.set_defaults(func=model_using_transition_probs)

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
        n_threads = args.n_threads
        pool = Pool(args.n_threads)
        logging.info("Made pool")
        model = None

        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
        genotype_matrix = GenotypeMatrix.from_file(args.genotype_matrix)
        # NB: Transpose
        genotype_matrix.matrix = genotype_matrix.matrix.transpose()

        logging.info("Genotype matrix shape: %s" % str(genotype_matrix.matrix.shape))
        # convert to format used in helper code
        # genotype_matrix = genotype_matrix.convert_to_other_format()
        logging.info(
            "Genotype matrix shape after conversion: %s"
            % str(genotype_matrix.matrix.shape)
        )
        most_similar = None
        if args.most_similar_variants is not None:
            most_similar = MostSimilarVariantLookup.from_file(
                args.most_similar_variants
            )

        if args.n_threads > 1:
            n_variants = len(variant_to_nodes.ref_nodes)
            n_threads = args.n_threads
            while n_variants < n_threads * 50 and n_threads > 2:
                n_threads -= 1
                logging.info("Lowered n threads to %d so that not too few variants are analysed together" % n_threads)

            intervals = [int(i) for i in np.linspace(0, n_variants, n_threads)]
            variant_intervals = [
                (from_id, to_id)
                for from_id, to_id in zip(intervals[0:-1], intervals[1:])
            ]
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
                create_helper_model_single_thread, zip(variant_intervals, repeat(args))
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
    subparser.add_argument(
        "-A", "--count-model", required=True, help="Node count model"
    )
    subparser.add_argument(
        "-o",
        "--out-file-name",
        required=True,
        help="Will write genotyped variants to this file",
    )
    subparser.add_argument("-x", "--tricky-variants", required=True)
    subparser.add_argument("-f", "--helper-model", required=True)
    subparser.add_argument("-F", "--helper-model-combo-matrix", required=True)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.set_defaults(func=make_index_bundle)

    def sample_node_counts_from_population(args):
        if args.n_threads > 0:
            logging.info("Creating pool to run in parallel")
            get_shared_pool(args.n_threads)

        logging.info("Reading graph")
        args.graph = from_file(args.graph)
        log_memory_usage_now("After reading graph")
        try:
            args.kmer_index = from_file(args.kmer_index)
        except KeyError:
            args.kmer_index = KmerIndex.from_file(args.kmer_index)
            args.kmer_index.convert_to_int32()
            args.kmer_index.remove_ref_offsets()  # not needed, will save us some memory

        log_memory_usage_now("After reading kmer index")

        from obgraph.haplotype_nodes import HaplotypeToNodesRagged
        if "disc" in args.haplotype_to_nodes:
            args.haplotype_to_nodes = DiscBackedHaplotypeToNodes.from_file(args.haplotype_to_nodes)
        else:
            args.haplotype_to_nodes = HaplotypeToNodesRagged.from_file(args.haplotype_to_nodes)

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

    subparser = subparsers.add_parser("sample_node_counts_from_population")
    subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-k", "--kmer-size", required=False, type=int, default=31)
    subparser.add_argument(
        "-H", "--haplotype-to-nodes", required=True
    )
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-t", "--n-threads", required=False, type=int, default=1)
    subparser.add_argument("-M", "--max-count", required=False, type=int, default=30)
    subparser.add_argument("-l", "--limit-to-n-individuals", required=False, type=int, default=0)
    subparser.set_defaults(func=sample_node_counts_from_population)


    def refine_sampling_model(args):
        model = from_file(args.sampling_model)
        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)

        models = [
            model.subset_on_nodes(variant_to_nodes.ref_nodes),
            model.subset_on_nodes(variant_to_nodes.var_nodes)
        ]
        logging.info("Filling missing data")
        for m in models:
            m.astype(np.float16)
            m.fill_empty_data()

        to_file(models, args.out_file_name)
        logging.info("Wrote refined model to %s" % args.out_file_name)

    subparser = subparsers.add_parser("refine_sampling_model")
    subparser.add_argument("-s", "--sampling_model", required=True)
    subparser.add_argument("-v", "--variant-to-nodes", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=refine_sampling_model)


    def node_counts_from_gaf_cmd(args):
        edge_mapping = pickle.load(open(args.edge_mapping, "rb"))
        node_counts = node_counts_from_gaf(
            args.gaf, edge_mapping, args.min_mapq, args.min_score, args.max_node_id
        )
        np.save(args.out_file_name, node_counts)
        logging.info("Node counts saved to %s" % args.out_file_name)

    subparser = subparsers.add_parser("node_counts_from_gaf")
    subparser.add_argument(
        "-g", "--gaf", required=True, help="Mapped reads in gaf format"
    )
    subparser.add_argument(
        "-m",
        "--edge-mapping",
        required=True,
        help="Mapping from vg graph edges to dummy nodes. Created when adding dummy nodes with obgraph add_indel_nodes",
    )
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-q", "--min-mapq", required=False, type=int, default=0)
    subparser.add_argument("-s", "--min-score", required=False, type=int, default=120)
    subparser.add_argument("-n", "--max_node_id", required=True, type=int)
    subparser.set_defaults(func=node_counts_from_gaf_cmd)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)
    remove_shared_memory_in_session()
