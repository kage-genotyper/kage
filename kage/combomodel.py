import logging
import numpy as np
from scipy.stats import nbinom, poisson, binom
from scipy.special import gamma, factorial, gammaln, logsumexp, hyp2f1, hyp1f1, hyperu, factorial
from itertools import combinations

class CountModel:
    error_rate=0.01

class MultiplePoissonModel(CountModel):
    def __init__(self, base_lambda, repeat_dist, certain_counts):
        self._base_lambda = base_lambda
        self._repeat_dist = repeat_dist
        self._certain_counts = certain_counts[:, None]
        self._n_variants = self._certain_counts.size
        self._max_duplicates = self._repeat_dist.shape[1]-1

    @staticmethod
    def calc_repeat_log_dist_fast(allele_frequencies):
        allele_frequencies = np.tile(allele_frequencies, 2)
        n_variants, n_duplicates = allele_frequencies.shape
        ns = np.arange(n_duplicates)
        repeat_dist = np.zeros((n_variants, n_duplicates+1))
        repeat_dist[:, 0] = 1
        for i, col in enumerate(allele_frequencies.T):
            repeat_dist[:, 1:] = (repeat_dist[:, :-1]*col[:, None]+repeat_dist[:, 1:]*(1-col[:, None]))
            repeat_dist[:, 0]*=(1-col)
        assert np.allclose(repeat_dist.sum(axis=1), 1), repeat_dist.sum(axis=1)
        return np.log(repeat_dist)

    @classmethod
    def from_counts(cls, base_lambda, certain_counts, allele_frequencies):
        repeat_dist = cls.calc_repeat_log_dist_fast(allele_frequencies)
        return cls(base_lambda, repeat_dist, 2*certain_counts)

    def logpmf(self, k, n_copies=1):
        assert k.shape == (self._n_variants, ), (k.shape, self._n_variants)
        rates = (self._certain_counts + n_copies + np.arange(self._max_duplicates+1)[None, :]+self.error_rate)*self._base_lambda
        log_probs = poisson.logpmf(k[:, None], rates)
        tot_probs = log_probs+self._repeat_dist
        return logsumexp(tot_probs, axis=1)

class NegativeBinomialModel(CountModel):
    def __init__(self, base_lambda, r, p, certain_counts):
        self._base_lambda = base_lambda
        self._r = r[:, None]
        self._p = p[:, None]
        self._certain_counts = certain_counts[:, None]

    @classmethod
    def from_counts(cls, base_lambda, p_sum, p_sq_sum, certain_counts):
        p_sum = p_sum*2
        p_sq_sum = p_sq_sum*2
        alpha = (p_sum)**2/(p_sum-p_sq_sum)
        beta = p_sum/(base_lambda*(p_sum-p_sq_sum))
        return cls(base_lambda, alpha, 1/(1+beta), 2*certain_counts)

    def logpmf(self, k, n_copies=1):
        k = k[:, None]
        mu = (n_copies+self._certain_counts+self.error_rate)*self._base_lambda
        r, p = (self._r, self._p)
        h = hyperu(r, r + k + 1, mu / p)
        invalid = (h==0) | (mu==0) | (p==0)
        result =  -r * np.log(p / (1 - p)) - mu + (r + k) * np.log(mu) - gammaln(k + 1) + np.log(h)
        return result.flatten()


class PoissonModel(CountModel):
    def __init__(self, base_lambda, expected_count):
        self._base_lambda = base_lambda
        self._expected_count = expected_count

    @classmethod
    def from_counts(cls, base_lambda, certain_counts, p_sum):
        return cls(base_lambda, (certain_counts+p_sum)*2)

    def logpmf(self, k, n_copies=1):
        return poisson.logpmf(k, (self._expected_count+n_copies+self.error_rate)*self._base_lambda)

class ComboModel(CountModel):
    def __init__(self, models, model_indexes):
        self._models = models
        self._model_indexes = model_indexes

    @classmethod
    def from_counts(cls, base_lambda, p_sum, p_sq_sum, do_gamma_calc, certain_counts, allele_frequencies):
        models = []
        model_indices = np.empty(certain_counts.size, dtype="int")
        multi_poisson_mask = ~do_gamma_calc
        models.append(
            MultiplePoissonModel.from_counts(base_lambda, certain_counts[multi_poisson_mask], allele_frequencies[multi_poisson_mask]))
        model_indices[multi_poisson_mask] = 0
        nb_mask = do_gamma_calc & (p_sum**2 <= (p_sum-p_sq_sum)*10)
        models.append(
            NegativeBinomialModel.from_counts(base_lambda, p_sum[nb_mask], p_sq_sum[nb_mask], certain_counts[nb_mask]))
        model_indices[nb_mask] = 1
        poisson_mask = do_gamma_calc & (~nb_mask)
        models.append(
            PoissonModel.from_counts(base_lambda, certain_counts[poisson_mask], p_sum[poisson_mask]))
        model_indices[poisson_mask] = 2
        return cls(models, model_indices)

    def logpmf(self, k, n_copies=1):
        logpmf = np.zeros(k.size)
        for i, model in enumerate(self._models):
            mask = (self._model_indexes == i)
            logpmf[mask] = model.logpmf(k[mask], n_copies)
        return logpmf
