import itertools
import logging
import random
from multiprocessing.pool import Pool
from obgraph.variant_to_nodes import VariantToNodes
from obgraph.genotype_matrix import GenotypeMatrix

import numpy as np
from shared_memory_wrapper import from_shared_memory, to_shared_memory
from shared_memory_wrapper.util import interval_chunks

from kage.models.helper_model import make_helper_model_from_genotype_matrix_and_node_counts, \
    make_helper_model_from_genotype_matrix


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
