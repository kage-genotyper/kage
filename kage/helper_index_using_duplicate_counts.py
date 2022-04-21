import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


class BinomialModel:

    error_rate = 0.5

    def __init__(self, expected_ref, expected_alt):
        self._expected_ref = 2 * np.asanyarray(expected_ref)
        self._expected_alt = 2 * np.asanyarray(expected_alt)
        self._n_variants = len(self._expected_ref)

    def predict(self, k1, k2):
        return np.argmax([self.logpmf(k1, k2, g) for g in (0, 1, 2)], axis=0)

    def score(self, k1, k2):
        ps = np.array([self.logpmf(k1, k2, g) for g in (0, 1, 2)])
        return ps - logsumexp(ps, axis=0, keepdims=True)

    def logpmf(self, k1, k2, genotype):
        k1 = np.asanyarray(k1)
        k2 = np.asanyarray(k2)
        total = k1 + k2
        ps = (self._expected_ref + 2 - genotype + self.error_rate) / (
            2 + self._expected_ref + self._expected_alt + 2 * self.error_rate
        )
        return binom.logpmf(k1, total, ps)


class PriorModel:
    def __init__(self, model, genotype_probs):
        self._model = model
        self._n_variants = self._model._n_variants

        self._genotype_frequencies = genotype_probs  #
        assert np.allclose(logsumexp(self._genotype_frequencies, axis=-1), 0)

    def predict(self, k1, k2):
        probs = [self.logpmf(k1, k2, g) for g in range(3)]
        return np.argmax(probs, axis=0)

    def score(self, k1, k2):
        unnormalized = np.array([self.logpmf(k1, k2, g) for g in [0, 1, 2]]).T
        return unnormalized - logsumexp(unnormalized, axis=-1, keepdims=True)

    @classmethod
    def from_genotype_matrix(cls, model, genotype_matrix):
        genotype_counts = (
            np.array([np.sum(genotype_matrix == k, axis=-1) for k in range(3)]).T + 0.1
        )
        genotype_probs = np.log(
            genotype_counts / genotype_counts.sum(axis=-1, keepdims=True)
        )
        return cls(model, genotype_probs)

    def logpmf(self, ref_counts, alt_counts, genotype):
        return (
            self._model.logpmf(ref_counts, alt_counts, genotype)
            + self._genotype_frequencies[:, genotype]
        )


def get_weighted_calc_func(score_func, weights, k=1):
    def weighted_score_func(count_matrix, offset):
        w = weights[:-offset] if offset > 0 else weights[-offset:]
        return score_func(count_matrix) + w * k

    return weighted_score_func


def get_prob_weights(k_r, k_a, genotype_probs):
    model = BinomialModel(k_r, k_a)
    prior_model = PriorModel(model, np.log((genotype_probs)))
    prob_correct = get_prob_correct(prior_model)
    return np.log(prob_correct)


def get_prob_correct(model):
    N = 10
    correct_probs = np.zeros(model._n_variants)
    p_sum = np.zeros_like(correct_probs)
    for k in range(N + 1):
        predicted = model.predict(k, N - k)
        for H in (0, 1, 2):
            p_k = np.exp(model.logpmf(k, N - k, H))
            p_sum += p_k
            correct_probs[predicted == H] += p_k[predicted == H]
    return correct_probs
