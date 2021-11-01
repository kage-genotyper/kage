import logging
import numpy as np

MAIN = -1
HELPER = -2
M = MAIN
H = HELPER


def make_helper_model_from_genotype_matrix(genotype_matrix, most_similar_variant_lookup=False, dummy_count=1, window_size=100):
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
    genotype_matrix = new_genotype_matrix

    logging.info("Finding best helper")

    if most_similar_variant_lookup is not None:
        logging.info("Making from most similar variant lookup")
        helpers = most_similar_variant_lookup.lookup_array
    else:
        logging.info("Making raw from genotype matrix with window size %d" % window_size)
        logging.info("Creating combined matrices")
        combined = create_combined_matrices(genotype_matrix, window_size)
        helpers = find_best_helper(combined, calc_likelihood)

    helper_counts = genotype_matrix[helpers] * 3
    flat_idx = genotype_matrix + helper_counts
    genotype_combo_matrix = np.array([(flat_idx == k).sum(axis=1) for k in range(9)]).T.reshape(-1, 3, 3) + dummy_count

    return helpers, genotype_combo_matrix


def create_combined_matrices(genotype_matrix, window_size):
    logging.info("Multiplying genotype matrix")
    helper = genotype_matrix*3
    result = []
    for i in range (1, window_size):
        if i % 50 == 0:
            logging.info("Window size now: %d/%d" % (i, window_size))
        flat_idx = genotype_matrix[i:]+helper[:-i]
        tmp = np.array([(flat_idx==k).sum(axis=1) for k in range(9)])
        tmp = (tmp.T).reshape(-1, 3, 3)
        result.append(tmp)
    result = result
    """variants, helper-g, main-g"""
    return result

def calc_likelihood(count_matrix):
    """
    Axis -2 is helper genotype
    Axis -1 is main genotype
    """
    count_matrix = count_matrix+1
    p = count_matrix/count_matrix.sum(axis=M, keepdims=True)
    return np.sum(count_matrix*np.log(p), axis=(M, H))
    
def calc_argmax(count_matrix):
    """
    Axis -2 is helper genotype
    Axis -1 is main genotype
    """
    return np.sum(np.max(count_matrix, axis=M), axis=-1)/count_matrix.sum(axis=(M, H))


def find_best_helper(combined, score_func):
    N = len(combined[0])+1
    best_idx, best_score = np.empty(N, dtype="int"), -np.inf*np.ones(N)
    for j, counts in enumerate(combined, 1):
        scores = score_func(counts)
        do_update = scores>best_score[j:]
        best_score[j:][do_update] = scores[do_update]
        best_idx[j:][do_update] = np.flatnonzero(do_update)
        rev_scores = score_func(counts.swapaxes(-2, -1))
        do_update = rev_scores>best_score[:-j]
        best_score[:-j][do_update] = rev_scores[do_update]
        best_idx[:-j][do_update] = np.flatnonzero(do_update)+j
    ### best_idx[i] is the variant that helps predict variant i the best
    return best_idx

