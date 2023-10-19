import logging
import time

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

from .paths import PathCreator
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix, GenotypeMatrix
from kage.models.helper_model import HelperVariants, CombinationMatrix
from .path_based_count_model import PathBasedMappingModelCreator
from kage.models.mapping_model import convert_model_to_sparse
from kage.util import log_memory_usage_now
import bionumpy as bnp
from ..preprocessing.variants import get_padded_variants_from_vcf, VariantStream, FilteredVariantStream, \
    FilteredOnMaxAllelesVariantStream
from kage.models.helper_model import make_helper_model_from_genotype_matrix
from multiprocessing import Process


def make_index(reference_file_name, vcf_file_name, out_base_name, k=31,
               modulo=20000033, variant_window=6,
               n_threads=16):
    """
    Makes all indexes and writes to an index bundle.
    """
    t_start = time.perf_counter()
    logging.info("Making graph")
    reference_sequences = bnp.open(reference_file_name).read()
    # vcf_variants are original vcf variants, needed when writing final vcf after genotyping
    # n_alleles_per_variant lets us convert genotypes on variants (which are biallelic) to multiallelic where necessary
    variant_stream = FilteredVariantStream.from_vcf_with_snps_indels_inside_svs_removed(vcf_file_name, sv_size_limit=50)
    variant_stream = FilteredOnMaxAllelesVariantStream(variant_stream, max_alleles=2**variant_window-1)
    variants, vcf_variants, n_alleles_per_original_variant = get_padded_variants_from_vcf(variant_stream,
                                                                                          reference_file_name,
                                                                                          True,
                                                                                          remove_indel_padding=False)
    assert len(variants) == np.sum(n_alleles_per_original_variant-1), f"{len(variants)} != {np.sum(n_alleles_per_original_variant-1)}"
    assert len(vcf_variants) == len(n_alleles_per_original_variant), f"{len(vcf_variants)} != {len(n_alleles_per_original_variant)}"

    logging.info("N biallelic variants: %d" % len(variants))
    logging.info("N original variants: %d" % len(vcf_variants))

    graph, node_mapping = make_multiallelic_graph(reference_sequences, variants)
    if np.max(node_mapping.n_alleles_per_variant) >= 2**variant_window:
        logging.warning("Some variants have more alleles than supported by current window size (%d)" % variant_window)
        possible_windows = [w for w in range(8) if 2**w >= np.max(node_mapping.n_alleles_per_variant)]
        if len(possible_windows) == 0:
            logging.error("Could not find a window large enough. There are too many alleles (max alleles on a variant is %d)" % np.max(node_mapping.n_alleles_per_variant))
            raise Exception("Too many alleles")
        else:
            variant_window = min(possible_windows)
            logging.warning("Increased window to %d" % variant_window)

    if len(graph.genome.sequence[-1]) < k:
        # pad genome
        logging.warning("Last variant is too close to end of the genome. Padding")
        graph.genome.pad_at_end(k)

    logging.info("Making haplotype matrix")
    #variant_stream = VariantStream.from_vcf(vcf_file_name, buffer_type=bnp.io.vcf_buffers.PhasedHaplotypeVCFMatrixBuffer, min_chunk_size=500000000)
    variant_stream = FilteredVariantStream.from_vcf_with_snps_indels_inside_svs_removed(vcf_file_name,
                                                                                        buffer_type=bnp.io.vcf_buffers.PhasedHaplotypeVCFMatrixBuffer,
                                                                                        min_chunk_size=500000000,
                                                                                        sv_size_limit=50
                                                                                        )
    variant_stream = FilteredOnMaxAllelesVariantStream(variant_stream, max_alleles=2**variant_window-1)

    haplotype_matrix_original_vcf = SparseHaplotypeMatrix.from_vcf(variant_stream)
    logging.info("N variants in original haplotype matrix: %d" % haplotype_matrix_original_vcf.data.shape[0])
    # this haplotype matrix may be multiallelic, convert to biallelic which will match "variants"
    biallelic_haplotype_matrix = haplotype_matrix_original_vcf.to_biallelic(n_alleles_per_original_variant)
    logging.info("N variants in biallelic haplotype matrix: %d" % biallelic_haplotype_matrix.data.shape[0])

    # Convert biallelic haplotype matrix to multiallelic
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    haplotype_matrix = biallelic_haplotype_matrix.to_multiallelic(n_alleles_per_variant)
    logging.info(f"{haplotype_matrix.n_variants} variants after converting to multiallelic")


    log_memory_usage_now("After graph")
    # Start seperate process for helper model
    helper_model_process, helper_model_result_name = make_helper_model_seperate_process(biallelic_haplotype_matrix)

    scorer = make_kmer_scorer_from_random_haplotypes(graph, haplotype_matrix, k, n_haplotypes=0, modulo=modulo * 10)
    assert np.all(scorer.values >= 0)

    log_memory_usage_now("After scorer")

    logging.info("Making paths")
    paths = PathCreator(graph,
                        window=variant_window,  # bigger windows to get more paths when multiallelic
                        #make_disc_backed=True,
                        make_graph_backed_sequences=True,
                        disc_backed_file_base_name=out_base_name,
                        use_new_allele_matrix=True
                        ).run(n_alleles_per_variant)

    logging.info("Made %d paths to cover variants" % (len(paths.paths)))

    variant_to_nodes = node_mapping.get_variant_to_nodes()
    signatures_chunk_size = 2000
    if len(variants) > 1000000:
        signatures_chunk_size = 4000
    signatures = get_signatures(k, paths, scorer, chunk_size=signatures_chunk_size, spacing=k//2)

    kmer_index = signatures.get_as_kmer_index(node_mapping=node_mapping, modulo=modulo, k=k)

    logging.info("Creating count model")
    log_memory_usage_now("Before creating count model")
    model_creator = PathBasedMappingModelCreator(graph, kmer_index,
                                                 haplotype_matrix,
                                                 k=k,
                                                 paths_allele_matrix=paths.variant_alleles,
                                                 window=variant_window-2,
                                                 max_count=20,
                                                 node_map=node_mapping,
                                                 n_nodes=len(variants)*2,
                                                 n_threads=n_threads)
    count_model = model_creator.run()

    # find tricky variants before count model is refined
    #tricky_variants = find_tricky_variants_with_count_model(signatures, count_model)
    logging.info("Finding tricky variants")
    log_memory_usage_now("Finding tricky variants")
    tricky_variants = find_tricky_variants_from_multiallelic_signatures(signatures, node_mapping.n_biallelic_variants)
    log_memory_usage_now("Finding tricky variants 2")
    tricky_alleles = find_tricky_ref_and_var_alleles_from_count_model(count_model, node_mapping)
    log_memory_usage_now("Finding tricky variants 3")

    from ..models.mapping_model import refine_sampling_model_noncli
    count_model = refine_sampling_model_noncli(count_model, variant_to_nodes)
    logging.info("Converting to sparse model")
    convert_model_to_sparse(count_model)

    #numpy_variants = NumpyVariants.from_vcf(vcf_no_genotypes_file_name)
    indexes = {
            "variant_to_nodes": variant_to_nodes,
            "count_model": count_model,
            "kmer_index": kmer_index,
            #"numpy_variants": numpy_variants,
            "tricky_variants": tricky_variants,
            "tricky_alleles": tricky_alleles,
            "vcf_header": bnp.open(vcf_file_name).read_chunk().get_context("header"),
            "vcf_variants": vcf_variants,
            "n_alleles_per_variant": n_alleles_per_original_variant
        }
    # helper model
    #helper_model, combo_matrix = make_helper_model(biallelic_haplotype_matrix)
    helper_model_process.join()
    helper_model, combo_matrix = object_from_shared_memory(helper_model_result_name)
    indexes["helper_variants"] = HelperVariants(helper_model)
    indexes["combination_matrix"] = CombinationMatrix(combo_matrix)


    index = IndexBundle(
        indexes
    )
    logging.info("Writing indexes to file")
    index.to_file(out_base_name, compress=False)
    paths.remove_tmp_files()

    logging.info("Making indexes took %.2f sec" % (time.perf_counter() - t_start))



def make_helper_model_seperate_process(biallelic_haplotype_matrix):
    result_name = str(id(biallelic_haplotype_matrix))
    logging.info("Making helper model in seperate process")
    p = Process(target=make_helper_model, args=(object_to_shared_memory(biallelic_haplotype_matrix), result_name))
    p.start()
    return p, result_name

def make_helper_model(biallelic_haplotype_matrix, write_to_result_shared_memory_name=None):
    t0 = time.perf_counter()
    if isinstance(biallelic_haplotype_matrix, str):
        biallelic_haplotype_matrix = object_from_shared_memory(biallelic_haplotype_matrix)
    genotype_matrix = GenotypeMatrix.from_haplotype_matrix(biallelic_haplotype_matrix)
    logging.info("Making helper model")
    helper_model, combo_matrix = make_helper_model_from_genotype_matrix(
        genotype_matrix.matrix, None, dummy_count=1.0, window_size=100)

    logging.info("Making helper model took %.2f sec" % (time.perf_counter() - t0))
    if write_to_result_shared_memory_name is not None:
        object_to_shared_memory((helper_model, combo_matrix), write_to_result_shared_memory_name)
        return

    return helper_model, combo_matrix

def make_index_cli(args):
    r = make_index(args.reference, args.vcf, args.out_base_name,
                      args.kmer_size, modulo=args.modulo,
                      variant_window=args.variant_window)
    remove_shared_memory_in_session()
    return r