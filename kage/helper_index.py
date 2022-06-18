import logging
import numpy as np
from .joint_distribution import create_combined_matrices
from .helper_index_using_duplicate_counts import (
    get_weighted_calc_func,
    get_prob_weights,
)

MAIN = -1
HELPER = -2
M = MAIN
H = HELPER


class HelperVariants:
    properties = {"helper_variants"}

    def __init__(self, helper_variants):
        self.helper_variants = helper_variants

    @classmethod
    def from_file(cls, file_name):
        return cls(np.load(file_name))


class CombinationMatrix:
    properties = {"matrix"}

    def __init__(self, matrix):
        self.matrix = matrix

    def __getitem__(self, variant_id):
        return self.matrix[variant_id]

    @classmethod
    def from_file(cls, file_name):
        return cls(np.load(file_name))


def calc_likelihood(count_matrix):
    dummy_count = 1
    count_matrix = count_matrix + dummy_count
    p = count_matrix / count_matrix.sum(axis=M, keepdims=True)

    # boost_matrix = np.ones((3, 3))
    # boost_matrix[1, 1] = 2
    # boost_matrix[2, 2] = 2

    # boost_matrix[:,1] = np.array([1, 3, 3])
    # boost_matrix[1:3, 1:3] = np.ones((2, 2)) * 2
    return (
        np.log(count_matrix[:, 0, 0])
        + np.log(count_matrix[:, 1, 1])
        + np.log(count_matrix[:, 2, 2])
    )
    return np.sum(
        np.log(np.max(count_matrix, axis=2) + 0.00001 - np.mean(count_matrix, axis=2)),
        axis=1,
    )
    # return np.sum(np.log(np.max(count_matrix, axis=2) - np.min(count_matrix, axis=2)), axis=1)
    # return np.sum(np.log(np.max(count_matrix, axis=2)), axis=1)
    # return np.sum(np.log(np.sum(count_matrix, axis=M)), axis=1)
    return np.sum(boost_matrix * count_matrix * np.log(p), axis=(M, H))
    # return np.sum(count_matrix*np.log(p), axis=(M, H))


def calc_argmax(count_matrix):
    return np.sum(np.max(count_matrix, axis=M), axis=-1) / count_matrix.sum(axis=(M, H))


def make_helper_model_from_genotype_matrix_and_node_counts(
    genotype_matrix, node_counts, variant_to_nodes, window_size=1000, dummy_count=10
):
    logging.info("Using dummy count scale %d" % dummy_count)
    genotype_matrix = genotype_matrix.matrix
    # print(genotype_matrix.matrix)
    # genotype_matrix = convert_genotype_matrix(old_genotype_matrix)
    nodes_tuple = (variant_to_nodes.ref_nodes, variant_to_nodes.var_nodes)
    expected_ref, expected_alt = (
        node_counts.certain[nodes] + node_counts.frequencies[nodes]
        for nodes in nodes_tuple
    )

    genotype_counts = np.array(
        [np.sum(genotype_matrix == i, axis=-1) for i in range(3)]
    ).T
    mean_genotype_counts = np.mean(genotype_counts, axis=0)
    mean_genotype_counts *= dummy_count / np.sum(mean_genotype_counts)
    genotype_counts = genotype_counts + mean_genotype_counts
    mean_genotype_counts = (np.tile(mean_genotype_counts, 3).reshape(3, 3) / 3)[
        None, ...
    ]

    genotype_probs = genotype_counts / genotype_counts.sum(axis=-1, keepdims=True)
    weights = get_prob_weights(expected_ref, expected_alt, genotype_probs)

    score_func = get_weighted_calc_func(calc_likelihood, weights, 0.4)
    return make_helper_model_from_genotype_matrix(
        genotype_matrix,
        None,
        score_func=score_func,
        dummy_count=mean_genotype_counts,
        window_size=window_size,
    )


def get_helper_posterior(genotype_combo_matrix, global_helper_weight=5):
    # logging.info("Dtype genotype combo matrix: %s" % genotype_combo_matrix.dtype)
    helper_sum = np.sum(genotype_combo_matrix, axis=M, keepdims=True)
    assert helper_sum[0].shape == (3, 1)
    global_helper_prior = (
        np.mean(helper_sum, axis=0, keepdims=True) + 1 / genotype_combo_matrix.shape[0]
    )  # + 0.1*np.array([182, 20, 13])[:, None]/215  # numbers based on real data
    # print("Global helper prior: \n%s" % global_helper_prior)
    # print("Helper sum: \n%s" % helper_sum)
    assert global_helper_prior.shape == (1, 3, 1)
    helper_posterior = (
        global_helper_prior / global_helper_prior.sum() * global_helper_weight
        + helper_sum
    )
    # helper_posterior = global_helper_prior   # global_helper_prior * global_helper_weight + helper_sum
    helper_posterior = helper_posterior / helper_posterior.sum(axis=H, keepdims=True)
    assert np.allclose(helper_posterior.sum(axis=H), 1), helper_posterior
    return helper_posterior


def get_population_priors(
    genotype_combo_matrix,
    weight=150,
    weight_diagonal=0,
    weight_left_column=0,
    weight_global=1,
):
    """n_variants x helper x main"""
    prior = np.eye(3) * weight_diagonal
    prior[:, 0] = weight_left_column  # going to 0/0 is high
    prior += weight_global
    #print("Weights added to population priors: \n%s" % prior)
    mean = np.sum(genotype_combo_matrix, axis=0) + prior
    #print("Population prior before weighted: \n%s" % mean)
    weighted = mean / mean.sum(axis=M, keepdims=True) * weight  # helper_sum*weight
    #print("")
    #print("Population prior after weighted: \n%s" % weighted)
    return weighted


def make_helper_model_from_genotype_matrix(
    genotype_matrix,
    most_similar_variant_lookup=False,
    dummy_count=1,
    score_func=calc_likelihood,
    window_size=1000,
):
    # genotype_matrix = convert_genotype_matrix(genotype_matrix)
    logging.info("Finding best helper")

    if most_similar_variant_lookup is not None:
        logging.info("Making from most similar variant lookup")
        helpers = most_similar_variant_lookup.lookup_array
    else:
        logging.info(
            "Making raw from genotype matrix with window size %d" % window_size
        )
        combined = create_combined_matrices(genotype_matrix, window_size)
        helpers = find_best_helper(
            combined,
            score_func,
            len(genotype_matrix),
            with_model=score_func != calc_likelihood,
        )

    helper_counts = genotype_matrix[helpers] * 3
    flat_idx = genotype_matrix + helper_counts

    genotype_combo_matrix = np.array(
        [(flat_idx == k).sum(axis=1) for k in range(9)]
    ).T.reshape(-1, 3, 3)
    # print("########")
    population_prior = get_population_priors(genotype_combo_matrix)
    helper_posterior = get_helper_posterior(genotype_combo_matrix)
    # print("Genotype combo matrix raw:")
    # print(genotype_combo_matrix[0])
    # print("Population prior: \n%s" % population_prior)
    # print("Helper posterior:\n%s", helper_posterior[0])
    #print("Combo matrix before population priors:\n%s" % genotype_combo_matrix)
    population_posterior = genotype_combo_matrix + population_prior
    #print("Combo matrix after population priors:\n%s" % population_posterior)
    population_posterior = (
        population_posterior
        / population_posterior.sum(axis=M, keepdims=True)
        * helper_posterior
    )
    # print("Genotype combo matrix posterior: ")
    # print(population_posterior[0])

    assert len(helpers) == genotype_matrix.shape[0]

    return helpers, population_posterior


def find_best_helper(combined, score_func, N, with_model=False):
    best_idx, best_score = np.zeros(N, dtype="int"), -np.inf * np.ones(N)
    for j, counts in enumerate(combined, 1):
        if j < 4:
            continue
        if j % 50 == 0:
            logging.info("Window %d" % j)
        scores = score_func(counts, j) if with_model else score_func(counts)
        do_update = scores > best_score[j:]
        best_score[j:][do_update] = scores[do_update]
        best_idx[j:][do_update] = np.flatnonzero(do_update)
        # reverse
        rev_scores = (
            score_func(counts.swapaxes(-2, -1), -j)
            if with_model
            else score_func(counts.swapaxes(-2, -1))
        )
        do_update = rev_scores >= best_score[:-j]
        best_score[:-j][do_update] = rev_scores[do_update]
        best_idx[:-j][do_update] = np.flatnonzero(do_update) + j
    return best_idx
