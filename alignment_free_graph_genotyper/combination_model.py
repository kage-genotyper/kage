
import logging
import numpy as np
from scipy.stats import nbinom, poisson
from scipy.special import gamma, factorial, gammaln, logsumexp, hyp2f1, hyp1f1, hyperu, factorial
from itertools import combinations
import matplotlib.pyplot as plt

COVERAGE = 15


class CombinationModelWithHistogram:
    """Do real calculation when few unceritain duplications, else use gamma distr"""
    error_rate = 0.02

    def __init__(self, base_lambda, r, p, repeat_dist, do_gamma_calc, certain_counts):
        """
        dim(repeat_dist): n_variants x (max_duplicates+1)
        dim(certain_counts): n_variants
        """
        self._base_lambda = base_lambda
        self._r = r[:, None]
        self._p = p[:, None]
        self._repeat_dist = repeat_dist  # n_variants x
        self._max_duplicates = self._repeat_dist.shape[1] - 1
        self._n_variants = self._repeat_dist.shape[0]
        self._certain_counts = certain_counts[:, None]
        self._do_gamma_calc = do_gamma_calc

    @staticmethod
    def calc_repeat_dist(allele_frequencies):
        n_variants, n_duplicates = allele_frequencies.shape
        ns = np.arange(n_duplicates)
        repeat_dist = np.zeros((n_variants, n_duplicates + 1))

        p = allele_frequencies
        q = (1 - allele_frequencies)
        repeat_dist[:, 0] = np.prod(q, axis=1)
        for n in range(1, n_duplicates + 1):
            f = factorial(n)
            for comb in combinations(ns, n):
                prob = repeat_dist[:, 0] * np.prod(p[:, comb], axis=1) / np.prod(q[:, comb], axis=1)
                repeat_dist[:, n] += prob
        assert repeat_dist.shape == (n_variants, n_duplicates + 1), (repeat_dist.shape, (n_variants, n_duplicates + 1))
        return repeat_dist

    @staticmethod
    def calc_repeat_log_dist(allele_frequencies):
        n_variants, n_duplicates = allele_frequencies.shape
        ns = np.arange(n_duplicates)
        repeat_dist = np.zeros((n_variants, n_duplicates + 1))

        p = np.log(allele_frequencies)
        q = np.log((1 - allele_frequencies))
        repeat_dist[:, 0] = np.sum(q, axis=1)
        for n in range(1, n_duplicates + 1):
            probs = [repeat_dist[:, 0] + np.sum(p[:, comb], axis=1) - np.sum(q[:, comb], axis=1)
                     for comb in combinations(ns, n)]
            # print(list(combinations(ns, n)))
            # print(probs)
            repeat_dist[:, n] = logsumexp(probs, axis=0)
        assert repeat_dist.shape == (n_variants, n_duplicates + 1), (repeat_dist.shape, (n_variants, n_duplicates + 1))
        assert np.allclose(np.sum(np.exp(repeat_dist), axis=1), 1), np.exp(repeat_dist[:5, :])
        return repeat_dist

    @classmethod
    def from_counts(cls, base_lambda, p_sum, p_sq_sum, do_gamma_calc, certain_counts, allele_frequencies):
        alpha = (p_sum) ** 2 / (p_sum - p_sq_sum)
        beta = p_sum / (base_lambda * (p_sum - p_sq_sum))
        repeat_dist = cls.calc_repeat_log_dist(allele_frequencies)
        return cls(base_lambda, alpha, 1 / (1 + beta), repeat_dist, do_gamma_calc, certain_counts)

    def logpmf(self, k, n_copies=1):
        k = np.asanyarray(k)
        assert k.shape == (self._n_variants,), (k.shape, (self._n_variants,))
        dist_pmf = self.dist_logpmf(k, n_copies)
        assert dist_pmf.shape == (self._n_variants,), (dist_pmf.shape, (self._n_variants,))
        gamma_pmf = self.gamma_logpmf(k, n_copies)
        # print(dist_pmf, gamma_pmf)
        assert gamma_pmf.shape == (self._n_variants,), (gamma_pmf.shape, (self._n_variants,))
        ret = np.where(self._do_gamma_calc, gamma_pmf, dist_pmf)
        assert ret.shape == (self._n_variants,), (ret.shape, (self._n_variants,))
        return ret

    def dist_pmf(self, k, n_copies=1):
        assert k.shape == (self._n_variants,), (k.shape, self._n_variants)
        rates = (self._certain_counts + n_copies + np.arange(self._max_duplicates + 1)[None,
                                                   :] + self.error_rate) * self._base_lambda
        assert rates.shape == (self._n_variants, self._max_duplicates + 1), (
        rates.shape, (self._n_variants, self._max_duplicates + 1))
        log_probs = poisson.logpmf(k[:, None], rates)
        tot_probs = np.exp(log_probs) * self._repeat_dist
        return tot_probs.sum(axis=1)

    def dist_logpmf(self, k, n_copies=1):
        assert k.shape == (self._n_variants,), (k.shape, self._n_variants)
        rates = (self._certain_counts + n_copies + np.arange(self._max_duplicates + 1)[None,
                                                   :] + self.error_rate) * self._base_lambda
        assert rates.shape == (self._n_variants, self._max_duplicates + 1), (
        rates.shape, (self._n_variants, self._max_duplicates + 1))
        log_probs = poisson.logpmf(k[:, None], rates)
        tot_probs = log_probs + self._repeat_dist
        return logsumexp(tot_probs, axis=1)

    @classmethod
    def from_p_sums(cls, base_lambda, p_sum, p_sq_sum):
        sum_is_sq_sum = p_sum == p_sq_sum
        alpha = (p_sum) ** 2 / (p_sum - p_sq_sum)
        beta = p_sum / (base_lambda * (p_sum - p_sq_sum))
        return cls(base_lambda, alpha, 1 / (1 + beta), sum_is_sq_sum, p_sum)

    def gamma_logpmf(self, k, n_copies=1):
        k = k[:, None]
        mu = (n_copies + self._certain_counts + self.error_rate) * self._base_lambda
        r, p = (self._r, self._p)
        result = -r * np.log(p / (1 - p)) - mu + (r + k) * np.log(mu) - gammaln(k + 1) + np.log(
            hyperu(r, r + k + 1, mu / p))
        assert result.shape == (self._n_variants, 1), (result.shape, (self._n_variants, 1))
        return result.flatten()

