import logging
import numpy as np
from .joint_distribution import create_combined_matrices
from .helper_index_using_duplicate_counts import get_weighted_calc_func, get_prob_weights

MAIN = -1
HELPER = -2
M = MAIN
H = HELPER


def calc_likelihood(count_matrix):
    count_matrix = count_matrix+1
    p = count_matrix/count_matrix.sum(axis=M, keepdims=True)
    return np.sum(count_matrix*np.log(p), axis=(M, H))
    
def calc_argmax(count_matrix):
    return np.sum(np.max(count_matrix, axis=M), axis=-1)/count_matrix.sum(axis=(M, H))

def convert_genotype_matrix(genotype_matrix):
    genotype_matrix = genotype_matrix.matrix.transpose()

    # genotypes are 1, 2, 3 (0 for unknown, 1 for homo ref, 2 for homo alt and 3 for hetero), we want 0, 1, 2 for homo alt, hetero, homo ref
    logging.info("Converting genotype matrix to format in helper model code")
    # 0, 1 => 2
    # 2 => 0
    # 3 => 1
    new_genotype_matrix = np.zeros_like(genotype_matrix)
    new_genotype_matrix[np.where(genotype_matrix == 0)] = -1
    new_genotype_matrix[np.where(genotype_matrix == 1)] = 0
    new_genotype_matrix[np.where(genotype_matrix == 2)] = 2
    new_genotype_matrix[np.where(genotype_matrix == 3)] = 1
    return new_genotype_matrix

def make_helper_model_from_genotype_matrix_and_node_counts(old_genotype_matrix, node_counts, variant_to_nodes, dummy_count=1):
    genotype_matrix = convert_genotype_matrix(old_genotype_matrix)
    nodes_tuple = (variant_to_nodes.ref_nodes, variant_to_nodes.var_nodes)

    expected_ref, expected_alt = (node_counts.certain[nodes]+node_counts.frequencies[nodes] for nodes in nodes_tuple)
    
    genotype_counts = np.array([np.sum(genotype_matrix==i, axis=-1) for i in range(3)]).T
    mean_genotype_counts = np.mean(genotype_counts, axis=0)
    mean_genotype_counts *= dummy_count/np.sum(mean_genotype_counts)
    genotype_counts = genotype_counts+mean_genotype_counts
    genotype_probs = genotype_counts/genotype_counts.sum(axis=-1, keepdims=True)
    weights = get_prob_weights(expected_ref, expected_alt, genotype_probs)
    score_func = get_weighted_calc_func(calc_likelihood, weights, 0.4)
    return make_helper_model_from_genotype_matrix(old_genotype_matrix, None, score_func=score_func, dummy_count=mean_genotype_counts*mean_genotype_counts[:, None])

def make_helper_model_from_genotype_matrix(genotype_matrix, most_similar_variant_lookup=False, dummy_count=1, score_func=calc_likelihood, window_size=1000):
    genotype_matrix = convert_genotype_matrix(genotype_matrix)
    logging.info("Finding best helper")

    if most_similar_variant_lookup is not None:
        logging.info("Making from most similar variant lookup")
        helpers = most_similar_variant_lookup.lookup_array
    else:
        logging.info("Making raw from genotype matrix")
        logging.info("Creating combined matrices")
        combined = create_combined_matrices(genotype_matrix, window_size)
        helpers = find_best_helper(combined, calc_likelihood, len(genotype_matrix))

    helper_counts = genotype_matrix[helpers] * 3
    flat_idx = genotype_matrix + helper_counts
    genotype_combo_matrix = np.array([(flat_idx == k).sum(axis=1) for k in range(9)]).T.reshape(-1, 3, 3) + dummy_count

    return helpers, genotype_combo_matrix

def find_best_helper(combined, score_func, N, with_model=False):
    best_idx, best_score = np.empty(N, dtype="int"), -np.inf*np.ones(N)
    for j, counts in enumerate(combined, 1):
        scores = score_func(counts, j) if with_model else score_func(counts)
        do_update = scores > best_score[j:]
        best_score[j:][do_update] = scores[do_update]
        best_idx[j:][do_update] = np.flatnonzero(do_update)
        rev_scores = score_func(counts.swapaxes(-2, -1), -j) if with_model else score_func(counts.swapaxes(-2, -1))
        do_update = rev_scores>best_score[:-j]
        best_score[:-j][do_update] = rev_scores[do_update]
        best_idx[:-j][do_update] = np.flatnonzero(do_update)+j
    return best_idx
