import logging
from obgraph.variant_to_nodes import VariantToNodes
from obgraph.numpy_variants import NumpyVariants
import numpy as np
from kage.indexing.index_bundle import IndexBundle
from kage.indexing.path_variant_indexing import find_tricky_variants_from_signatures, find_tricky_variants_from_signatures2, find_tricky_variants_with_count_model, \
    MappingModelCreator
from kage.indexing.kmer_scoring import make_kmer_scorer_from_random_haplotypes
from kage.indexing.signatures import SignatureFinder3, MultiAllelicSignatureFinder, MultiAllelicSignatureFinderV2, \
    MatrixVariantWindowKmers, VariantWindowKmers2
from kage.indexing.graph import Graph, make_multiallelic_graph
from .paths import PathCreator
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix, GenotypeMatrix
from kage.models.helper_model import HelperVariants, CombinationMatrix
from .path_based_count_model import PathBasedMappingModelCreator
from kage.models.mapping_model import convert_model_to_sparse
from kage.util import log_memory_usage_now
import bionumpy as bnp
from ..preprocessing.variants import get_padded_variants_from_vcf


def make_index(reference_file_name, vcf_file_name, vcf_no_genotypes_file_name, out_base_name, k=31,
               modulo=20000033, variant_window=3, make_helper_model=False):
    """
    Makes all indexes and writes to an index bundle.
    """
    logging.info("Making graph")
    reference_sequences = bnp.open(reference_file_name).read()
    variants = get_padded_variants_from_vcf(vcf_file_name, reference_file_name)
    logging.info("N variants: %d" % len(variants))
    #graph = Graph.from_variants_and_reference(reference_sequences, variants)
    # todo: Make multiallelic graph instead, get node mapping
    graph, node_mapping = make_multiallelic_graph(reference_sequences, variants)
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    if len(graph.genome.sequence[-1]) < k:
        # pad genome
        logging.warning("Last variant is too close to end of the genome. Padding")
        graph.genome.pad_at_end(k)

    logging.info("Making haplotype matrix")
    biallelic_haplotype_matrix = SparseHaplotypeMatrix.from_vcf(vcf_file_name)
    logging.info("N variants in haplotype matrix: %d" % biallelic_haplotype_matrix.data.shape[0])
    haplotype_matrix = biallelic_haplotype_matrix.to_multiallelic(n_alleles_per_variant)
    logging.info(f"{haplotype_matrix.n_variants} variants after converting to multiallelic")

    log_memory_usage_now("After graph")
    scorer = make_kmer_scorer_from_random_haplotypes(graph, haplotype_matrix, k, n_haplotypes=4, modulo=modulo)

    log_memory_usage_now("After scorer")

    logging.info("Making paths")
    # todo: Run with n_alleles_per_variant to get multiallelic paths
    paths = PathCreator(graph,
                        window=variant_window+1,  # bigger windows to get more paths when multiallelic
                        make_disc_backed=True,
                        disc_backed_file_base_name=out_base_name).run(n_alleles_per_variant)

    # todo: Get variant_to_nodes from node_mapping instead
    #variant_to_nodes = VariantToNodes(np.arange(graph.n_variants())*2, np.arange(graph.n_variants())*2+1)
    variant_to_nodes = node_mapping.get_variant_to_nodes()

    # todo: Change to MultiAllelicSignatureFinder
    #Old:
    #signatures = SignatureFinder3(paths, scorer=scorer, k=k).run(variants)
    # New multiallelic
    #signatures = MultiAllelicSignatureFinder(paths, scorer=scorer, k=k).run()

    # New: Using FinderV2 which is faster
    logging.info("Finding kmers around variants")
    variant_window_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(paths.paths, k)
    logging.info("Converting variant window kmers to new data structure")
    variant_window_kmers = VariantWindowKmers2.from_matrix_variant_window_kmers(variant_window_kmers, paths.variant_alleles.matrix)
    logging.info("Finding best signatures for variants")
    signatures = MultiAllelicSignatureFinderV2(variant_window_kmers, scorer=scorer, k=k).run()

    # todo: Send in node_mapping to get kmer_index with correct node ids
    kmer_index = signatures.get_as_kmer_index(node_mapping=node_mapping, modulo=modulo, k=k)

    logging.info("Creating count model")
    logging.info("N nodes: %d" % node_mapping.n_nodes)
    model_creator = PathBasedMappingModelCreator(graph, kmer_index,
                                                 haplotype_matrix,
                                                 k=k,
                                                 paths_allele_matrix=paths.variant_alleles,
                                                 window=variant_window,
                                                 max_count=20,
                                                 node_map=node_mapping,
                                                 n_nodes=len(variants)*2)
    count_model = model_creator.run()

    # find tricky variants before count model is refined
    tricky_variants = None  # find_tricky_variants_with_count_model(signatures, count_model)

    from ..models.mapping_model import refine_sampling_model_noncli
    count_model = refine_sampling_model_noncli(count_model, variant_to_nodes)
    convert_model_to_sparse(count_model)

    numpy_variants = NumpyVariants.from_vcf(vcf_no_genotypes_file_name)
    indexes = {
            "variant_to_nodes": variant_to_nodes,
            "count_model": count_model,
            "kmer_index": kmer_index,
            "numpy_variants": numpy_variants,
            #"tricky_variants": tricky_variants,
        }
    # helper model
    if make_helper_model:
        from kage.models.helper_model import make_helper_model_from_genotype_matrix
        genotype_matrix = GenotypeMatrix.from_haplotype_matrix(biallelic_haplotype_matrix)
        logging.info("Making helper model")
        helper_model, combo_matrix = make_helper_model_from_genotype_matrix(
            genotype_matrix.matrix, None, dummy_count=1.0, window_size=100)
        indexes["helper_variants"] = HelperVariants(helper_model)
        indexes["combination_matrix"] = CombinationMatrix(combo_matrix)
    else:
        logging.info("Not making helper model")


    index = IndexBundle(
        indexes
    )
    index.to_file(out_base_name, compress=False)


def make_index_cli(args):
    return make_index(args.reference, args.vcf, args.vcf_no_genotypes, args.out_base_name,
                      args.kmer_size, make_helper_model=args.make_helper_model, modulo=args.modulo,
                      variant_window=args.variant_window)
