import logging
import sys

from kage.indexing.tricky_variants import TrickyVariants

logging.basicConfig(level=logging.INFO)
import time
import npstructures as nps
from dataclasses import dataclass
from obgraph.numpy_variants import NumpyVariants
import numpy as np
from kage.indexing.index_bundle import IndexBundle
from kage.indexing.path_variant_indexing import find_tricky_variants_from_multiallelic_signatures, \
    find_tricky_ref_and_var_alleles_from_count_model
from kage.indexing.kmer_scoring import make_kmer_scorer_from_random_haplotypes
from kage.indexing.signatures import get_signatures
from kage.indexing.graph import make_multiallelic_graph
from shared_memory_wrapper import to_file, remove_shared_memory_in_session, object_to_shared_memory, from_shared_memory, \
    object_from_shared_memory

from .paths import PathCreator, PathCombinationMatrix
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix, GenotypeMatrix
from kage.models.helper_model import HelperVariants, CombinationMatrix, \
    get_variants_that_can_be_used_as_helper_variants, get_variants_with_perfect_model
from .path_based_count_model import PathBasedMappingModelCreator
from kage.models.mapping_model import convert_model_to_sparse
from kage.util import log_memory_usage_now
import bionumpy as bnp
from ..models.mapping_model import refine_sampling_model_noncli

from ..preprocessing.variants import get_padded_variants_from_vcf, VariantStream, FilteredVariantStream, \
    FilteredOnMaxAllelesVariantStream, FilteredOnMaxAllelesVariantStream2, filter_variants_with_more_alleles_than, \
    LowAfDeletionsReplacedVariantStream
from kage.models.helper_model import make_helper_model_from_genotype_matrix
from multiprocessing import Process


def validate_input_vcf(vcf_file_name):
    logging.info("Validating input vcf")
    with bnp.open(vcf_file_name) as f:
        for chunk in f:
            if np.any(chunk.alt_seq == ","):
                logging.error("VCF contains multiallelic variants. Please split multiallelic variants, e.g. using "
                              "bcftools norm -m -any <input.vcf> before running kage index")
                sys.exit()

            if np.any(chunk.alt_seq == "<") or np.any(chunk.alt_seq == "*"):
                logging.error("Input VCF needs to contain only A, C, G, T or N in reference and alt sequences. Some alleles did not.")
                sys.exit()

            if not hasattr(chunk.info, "AF"):
                logging.error("VCF needs to contain the AF field in the INFO column (allele frequency used internally "
                              "by KAGE")
                logging.error("Please add AF to the INFO column, e.g. by using  bcftools +fill-tags in.vcf -Ob -o "
                              "out.vcf -- -t AF")
                sys.exit()

def make_index(
       reference_file_name,
       vcf_file_name,
       out_base_name,
       k=31,
       modulo=20000033,
       variant_window=6,
       n_threads=16,
       vcf_no_genotypes=None,
       min_af_deletions_filter=0.1,
       max_paths: int=100,
       no_helper_model=False
):
    """
    Makes all indexes and writes to an index bundle.
    """
    validate_input_vcf(vcf_file_name)
    t_start = time.perf_counter()
    # vcf_variants are original vcf variants, needed when writing final vcf after genotyping
    # n_alleles_per_variant lets us convert genotypes on variants (which are biallelic) to multiallelic where necessary
    if vcf_no_genotypes is None:
        vcf_no_genotypes = vcf_file_name
    variant_stream = FilteredVariantStream.from_vcf_with_snps_indels_inside_svs_removed(vcf_file_name,
                                                                                        min_chunk_size=500_000_000,
                                                                                        sv_size_limit=50,
                                                                                        buffer_type=bnp.io.vcf_buffers.VCFBuffer,
                                                                                        filter_using_other_vcf=vcf_no_genotypes)

    # convert variants to biallelic variants, keep track of how many alleles original variants had
    variants, vcf_variants, n_alleles_per_original_variant = get_padded_variants_from_vcf(variant_stream,
                                                                                          reference_file_name,
                                                                                          True,
                                                                                          remove_indel_padding=False,
                                                                                          remove_sequence_from_low_af_deletions=min_af_deletions_filter)

    variants.replace_ns()
    n_orig_variants_before_filtering = len(vcf_variants)

    # find variants with more alleles (use "variants" which are padded biallelic and look for variants starting
    # at same position).
    # filter away variants with too many alleles and filte vcf_variants and n_alleles_per_original_variant accordingly
    # Keep a filter of what has been filtered and use that filter later when getting haplotype matrix
    #filter_on_max_alleles = n_alleles_per_original_variant >= 2**variant_window-2
    variants, vcf_variants, n_alleles_per_original_variant, filter = filter_variants_with_more_alleles_than(variants,
                                                                                                            vcf_variants,
                                                                                                            n_alleles_per_original_variant,
                                                                                                            max_paths)
                                                                                                            #2**variant_window-2)
    log_memory_usage_now("After getting variants")
    assert len(filter) == n_orig_variants_before_filtering
    assert len(variants) == np.sum(n_alleles_per_original_variant-1), f"{len(variants)} != {np.sum(n_alleles_per_original_variant-1)}"
    assert len(vcf_variants) == len(n_alleles_per_original_variant), f"{len(vcf_variants)} != {len(n_alleles_per_original_variant)}"

    logging.info("N biallelic variants: %d" % len(variants))
    logging.info("N original variants: %d" % len(vcf_variants))
    logging.info("Max alleles on original variant: %d" % np.max(n_alleles_per_original_variant))

    logging.info("Making graph")
    reference_sequences = bnp.open(reference_file_name).read()
    log_memory_usage_now("After reading reference genome")
    graph, node_mapping = make_multiallelic_graph(reference_sequences, variants)
    del reference_sequences
    log_memory_usage_now("Made graph")

    if np.max(node_mapping.n_alleles_per_variant) >= 2**variant_window:
        logging.warning("Some variants have more alleles than supported by current window size (%d)" % variant_window)
        raise Exception("Something went wrong.")

    if len(graph.genome.sequence[-1]) < k:
        # pad genome
        logging.warning("Last variant is too close to end of the genome. Padding")
        graph.genome.pad_at_end(k)

    logging.info("Making haplotype matrix")
    variant_stream = FilteredVariantStream.from_vcf_with_snps_indels_inside_svs_removed(vcf_file_name,
                                                                                        buffer_type=bnp.io.vcf_buffers.PhasedHaplotypeVCFMatrixBuffer,
                                                                                        min_chunk_size=500_000_000,
                                                                                        sv_size_limit=50,
                                                                                        filter_using_other_vcf=vcf_no_genotypes
                                                                                        )
    log_memory_usage_now("Variant stream 1 done")
    variant_stream = FilteredVariantStream(variant_stream.read_chunks(), ~filter)
    log_memory_usage_now("Done variant stream")

    haplotype_matrix_original_vcf = SparseHaplotypeMatrix.from_vcf(variant_stream)
    log_memory_usage_now("Made haplotype matrix orig vcf")
    logging.info("N variants in original haplotype matrix: %d" % haplotype_matrix_original_vcf.data.shape[0])
    # this haplotype matrix may be multiallelic, convert to biallelic which will match "variants"
    biallelic_haplotype_matrix = haplotype_matrix_original_vcf.to_biallelic(n_alleles_per_original_variant)
    log_memory_usage_now("Made biallelic haplotype matrix")
    logging.info("N variants in biallelic haplotype matrix: %d" % biallelic_haplotype_matrix.data.shape[0])

    # Convert biallelic haplotype matrix to multiallelic
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    haplotype_matrix = biallelic_haplotype_matrix.to_multiallelic(n_alleles_per_variant)
    logging.info(f"{haplotype_matrix.n_variants} variants after converting to multiallelic")

    log_memory_usage_now("After graph")

    scorer = make_kmer_scorer_from_random_haplotypes(graph, haplotype_matrix, k, n_haplotypes=8, modulo=modulo * 40 + 11)
    assert np.all(scorer.values >= 0)

    log_memory_usage_now("After scorer")

    logging.info("Making paths")
    # use haplotype matrix as combination matrix, should work well when number of haplotypes <= number of paths
    combination_matrix = PathCombinationMatrix.from_sparse_haplotype_matrix(haplotype_matrix)
    combination_matrix.add_permuted_paths(n_alleles_per_variant, 3)
    combination_matrix.add_paths_with_missing_alleles_by_changing_existing_paths(n_alleles_per_variant)
    combination_matrix.add_paths_with_missing_alleles(n_alleles_per_variant)
    combination_matrix.sanity_check()
    combination_matrix.assert_all_alleles_are_supported(n_alleles_per_variant)

    paths = PathCreator(graph,
                        window=variant_window,  # bigger windows to get more paths when multiallelic
                        make_graph_backed_sequences=True,
                        disc_backed_file_base_name=out_base_name,
                        use_new_allele_matrix=True
                        ).run(n_alleles_per_variant, with_combination_matrix=combination_matrix)

    logging.info("Made %d paths to cover variants" % (len(paths.paths)))

    variant_to_nodes = node_mapping.get_variant_to_nodes()

    signatures_chunk_size = 500

    signatures = get_signatures(k, paths, scorer, chunk_size=signatures_chunk_size, spacing=0,
                                minimum_overlap_with_variant=2, n_threads=n_threads)

    kmer_index = signatures.get_as_kmer_index(node_mapping=node_mapping, modulo=modulo, k=k)
    logging.info("Kmer index contains %s kmers" % len(kmer_index.get_kmers()))

    logging.info("Creating count model")
    log_memory_usage_now("Before creating count model")
    model_creator = PathBasedMappingModelCreator(graph, kmer_index,
                                                 haplotype_matrix,
                                                 k=k,
                                                 paths_allele_matrix=paths.variant_alleles,
                                                 window=4,
                                                 max_count=20,
                                                 node_map=node_mapping,
                                                 n_nodes=len(variants)*2,
                                                 n_threads=n_threads)
    count_model = model_creator.run()

    # find tricky variants before count model is refined

    log_memory_usage_now("Finding tricky variants")
    tricky_variants, tricky_ref1, tricky_alt1 = find_tricky_variants_from_multiallelic_signatures(signatures, node_mapping.n_biallelic_variants, also_find_tricky_alleles=True)
    logging.info(f"{np.sum(tricky_ref1.tricky_variants)} tricky ref alleles because no kmers")
    logging.info(f"{np.sum(tricky_alt1.tricky_variants)} tricky alt alleles because no kmers")
    tricky_ref, tricky_alt = find_tricky_ref_and_var_alleles_from_count_model(count_model, node_mapping, max_count=7)

    tricky_ref.add(tricky_ref1)
    tricky_alt.add(tricky_alt1)
    logging.info(f"{np.sum(tricky_ref.tricky_variants)} total tricky ref alleles")
    logging.info(f"{np.sum(tricky_alt.tricky_variants)} total tricky alt alleles")

    tricky_alleles = (tricky_ref, tricky_alt)

    refined_count_model = refine_sampling_model_noncli(count_model, variant_to_nodes, prior_empty_data=0.1)

    # Start seperate process for helper model
    if not no_helper_model:
        #only_consider_variants_for_helper_model = get_variants_that_can_be_used_as_helper_variants(n_alleles_per_variant)
        only_consider_variants_for_helper_model = np.ones(np.sum(n_alleles_per_variant - 1), dtype=bool)
        #only_consider_variants_for_helper_model &= ~tricky_ref.tricky_variants
        #only_consider_variants_for_helper_model &= ~tricky_alt.tricky_variants
        ##has_perfect_model = get_variants_with_perfect_model(*refined_count_model)
        ##tricky_variants.add(TrickyVariants(~has_perfect_model))
        helper_model_process, helper_model_result_name = make_helper_model_seperate_process(biallelic_haplotype_matrix, only_consider_variants_for_helper_model)
    del biallelic_haplotype_matrix
    logging.info("Converting to sparse model")
    convert_model_to_sparse(refined_count_model)

    indexes = {
            "variant_to_nodes": variant_to_nodes,
            "count_model": refined_count_model,
            "kmer_index": kmer_index,
            "tricky_variants": tricky_variants,
            "tricky_alleles": tricky_alleles,
            "vcf_header": bnp.open(vcf_file_name).read_chunk().get_context("header"),
            "vcf_variants": vcf_variants,
            "n_alleles_per_variant": n_alleles_per_original_variant,
            "multiallelic_map": MultiAllelicMap.from_n_alleles_per_variant(n_alleles_per_original_variant),
            "orig_count_model": count_model
        }

    if not no_helper_model:
        # helper model
        #helper_model, combo_matrix = make_helper_model(biallelic_haplotype_matrix)
        helper_model_process.join()
        helper_model, combo_matrix = object_from_shared_memory(helper_model_result_name)
        indexes["helper_variants"] = HelperVariants(helper_model)
        indexes["combination_matrix"] = CombinationMatrix(combo_matrix)


    index = IndexBundle(indexes)
    logging.info("Writing indexes to file")
    index.to_file(out_base_name, compress=False)
    paths.remove_tmp_files()

    logging.info("Making indexes took %.2f sec" % (time.perf_counter() - t_start))
    return index, signatures, count_model


def make_helper_model_seperate_process(biallelic_haplotype_matrix, variant_filter):
    result_name = str(id(biallelic_haplotype_matrix))
    logging.info("Making helper model in seperate process")
    p = Process(target=make_helper_model,
                args=(object_to_shared_memory(biallelic_haplotype_matrix),
                      object_to_shared_memory(variant_filter),
                      result_name))
    p.start()
    return p, result_name


def make_helper_model(biallelic_haplotype_matrix, variant_filter, write_to_result_shared_memory_name=None):
    t0 = time.perf_counter()
    if isinstance(biallelic_haplotype_matrix, str):
        biallelic_haplotype_matrix = object_from_shared_memory(biallelic_haplotype_matrix)
    if isinstance(variant_filter, str):
        variant_filter = object_from_shared_memory(variant_filter)
    genotype_matrix = GenotypeMatrix.from_haplotype_matrix(biallelic_haplotype_matrix)
    logging.info("Making helper model")
    helper_model, combo_matrix = make_helper_model_from_genotype_matrix(
        genotype_matrix.matrix, None, dummy_count=1.0, window_size=100,
        only_consider_variants=variant_filter
    )

    logging.info("Making helper model took %.2f sec" % (time.perf_counter() - t0))
    if write_to_result_shared_memory_name is not None:
        object_to_shared_memory((helper_model, combo_matrix), write_to_result_shared_memory_name)
        return

    return helper_model, combo_matrix

def make_index_cli(args):
    r = make_index(args.reference, args.vcf, args.out_base_name,
                    args.kmer_size, modulo=args.modulo,
                    variant_window=args.variant_window,
                    vcf_no_genotypes=args.vcf_no_genotypes,
                    min_af_deletions_filter=args.min_af_deletions_filter,
                    no_helper_model=args.no_helper_model,
                    n_threads=args.n_threads,
                   )
    remove_shared_memory_in_session()
    return r



@dataclass
class MultiAllelicMap:
    """
    Data is a ragged array where ids of variants that are part of same
    multiallelic variant are on the same row
    """
    data: nps.RaggedArray

    @classmethod
    def from_n_alleles_per_variant(cls, n_alleles_per_variant):
        assert np.all(n_alleles_per_variant >= 2)
        tot = np.sum(n_alleles_per_variant-1)
        return nps.RaggedArray(np.arange(tot), n_alleles_per_variant-1)

    @classmethod
    def from_variants_by_position(cls, vcf_entry):
        n_alleles = []
        prev_chrom = ""
        prev_pos = 0
        for variant in vcf_entry:
            chrom = variant.chromosome.to_string()
            pos = int(variant.position)

            if chrom != prev_chrom or pos != prev_pos:
                n_alleles.append(2)
            else:
                n_alleles[-1] += 1

            prev_chrom = chrom
            prev_pos = pos

        return cls.from_n_alleles_per_variant(np.array(n_alleles))