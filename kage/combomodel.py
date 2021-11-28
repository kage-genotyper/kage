import logging
import time
import gc
from graph_kmer_index.shared_mem import run_numpy_based_function_in_parallel
from .util import log_memory_usage_now

import numpy as np
from scipy.stats import nbinom, poisson, binom
from scipy.special import gamma, factorial, gammaln, logsumexp, hyp2f1, hyp1f1, hyperu, factorial
from itertools import combinations


def parallel_poisson_logpmf(k, rates):
    return run_numpy_based_function_in_parallel(poisson.logpmf, 16, [k, rates])

def logsumexp_wrappper(probs):
    return logsumexp(probs, axis=1)

def parallel_logsumexp(probs):
    return run_numpy_based_function_in_parallel(logsumexp_wrappper, 16, [probs])


class CountModel:
    error_rate = 0.01

class MultiplePoissonModel(CountModel):
    def __init__(self, base_lambda, repeat_dist, certain_counts):
        self._base_lambda = base_lambda
        self._repeat_dist = repeat_dist.astype(np.float32)
        self._certain_counts = certain_counts[:, None].astype(np.uint16)
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

    @staticmethod
    def calc_repeat_log_dist_fast_parallel(allele_frequencies, n_threads=16):
        return run_numpy_based_function_in_parallel(MultiplePoissonModel.calc_repeat_log_dist_fast, n_threads, [allele_frequencies])

    @classmethod
    def from_counts(cls, base_lambda, certain_counts, allele_frequencies):
        repeat_dist = cls.calc_repeat_log_dist_fast_parallel(allele_frequencies)
        return cls(base_lambda, repeat_dist, 2*certain_counts)

    def logpmf(self, k, n_copies=1):
        assert k.shape == (self._n_variants, ), (k.shape, self._n_variants)
        t = time.perf_counter()
        rates = (self._certain_counts + n_copies + np.arange(self._max_duplicates+1)[None, :]+self.error_rate)*self._base_lambda
        logging.debug("Time getting rates: %.4f" % (time.perf_counter()-t))
        t = time.perf_counter()
        #log_probs = poisson.logpmf(k[:, None], rates)
        log_probs = parallel_poisson_logpmf(k[:, None], rates)
        logging.debug("Time poisson logpmf: %.4f" % (time.perf_counter()-t))
        t = time.perf_counter()
        tot_probs = log_probs+self._repeat_dist
        logging.debug("Time tot probs: %.4f" % (time.perf_counter()-t))
        t = time.perf_counter()
        #result = logsumexp(tot_probs, axis=1)
        result = parallel_logsumexp(tot_probs)
        logging.debug("Time logsumexp: %.4f" % (time.perf_counter()-t))
        return result


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
    def __init__(self, models, model_indexes, tricky_variants=None):
        self._models = models
        self._model_indexes = model_indexes
        self._tricky_variants = tricky_variants

    @classmethod
    def from_counts(cls, base_lambda, p_sum, p_sq_sum, do_gamma_calc, certain_counts, allele_frequencies):
        models = []
        model_indices = np.empty(certain_counts.size, np.uint8)
        t = time.perf_counter()
        multi_poisson_mask = ~do_gamma_calc
        t = time.perf_counter()
        models.append(
            MultiplePoissonModel.from_counts(base_lambda, certain_counts[multi_poisson_mask], allele_frequencies[multi_poisson_mask]))
        model_indices[multi_poisson_mask] = 0
        logging.debug("Time spent on creating MultiplePoisson model: %.3f" % (time.perf_counter()-t))
        t = time.perf_counter()

        nb_mask = do_gamma_calc & (p_sum**2 <= (p_sum-p_sq_sum)*10)
        models.append(
            NegativeBinomialModel.from_counts(base_lambda, p_sum[nb_mask], p_sq_sum[nb_mask], certain_counts[nb_mask]))
        model_indices[nb_mask] = 1
        logging.debug("Time spent on creating negative binomial model: %.3f" % (time.perf_counter()-t))
        t = time.perf_counter()

        poisson_mask = do_gamma_calc & (~nb_mask)
        models.append(
            PoissonModel.from_counts(base_lambda, certain_counts[poisson_mask], p_sum[poisson_mask]))
        model_indices[poisson_mask] = 2
        logging.debug("Time spent on creating Pisson model: %.3f" % (time.perf_counter()-t))
        t = time.perf_counter()

        return cls(models, model_indices)

    def logpmf(self, k, n_copies=1):
        logpmf = np.zeros(k.size, dtype=np.float16)
        for i, model in enumerate(self._models):
            mask = (self._model_indexes == i)
            start_time = time.perf_counter()
            logpmf[mask] = model.logpmf(k[mask], n_copies).astype(np.float16)
            logging.debug("Time spent on ComboModel.logpmf %s: %.4f" % (model.__class__, time.perf_counter()-start_time))
            gc.collect()

        # adjust with prob of counts being wrong
        p_counts_are_wrong = 0
        logpmf = np.logaddexp(np.log(p_counts_are_wrong) + np.log(1/3), np.log(1-p_counts_are_wrong) + logpmf)

        return logpmf.astype(np.float16)


class ComboModelParallel(CountModel):
    """Same interface as ComboModel, but has N ComboModels internally, which runs in parallel"""
    def __init__(self, combo_models):
        pass

    @classmethod
    def from_counts(cls, base_lambda, p_sum, p_sq_sum, do_gamma_calc, certain_counts, allele_frequencies):
        pass

    def logpmf(self, k, n_copies=1):
        pass


class ComboModelWithIncreasedZeroProb(ComboModel):
    def logpmf(self, k, n_copies=1):
        p_variant_has_zero_counts = 0.005
        logging.info("Using p variant has zero counts: %.5f" % p_variant_has_zero_counts)
        p = super().logpmf(k, n_copies)

        # subtract prob of variant having zero where k == 0, add 1- to all
        p = p - np.log(1 - p_variant_has_zero_counts)
        print(p)
        return np.where(k == 0, np.logaddexp(p, np.log(p_variant_has_zero_counts)), p)





