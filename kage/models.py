import time
from shared_memory_wrapper.shared_memory import run_numpy_based_function_in_parallel
import numpy as np
from scipy.special import logsumexp
import logging
from .util import log_memory_usage_now

MAIN = -1
HELPER = -2
M = MAIN
H = HELPER

class Model:
    def predict(self, k1, k2, return_probs=False):
        scores = self.score(k1, k2)
        result = np.argmax(scores, axis=-1)

        if return_probs:
            return result, scores

    def score(self, k1, k2):
        return np.array([self.logpmf(k1, k2, g) for g in [0, 1, 2]]).T

    def logpmf(self, k1, k2, genotype):
        raise NotImplemented

class ComboModelBothAlleles(Model):
    def __init__(self, model_ref, model_alt):
        self._model_ref = model_ref
        self._model_alt = model_alt
        self._logpmf_cache = {}

    def compute_logpmfs(self, k1, k2):
        for genotype in [0, 1, 2]:
            self.logpmf(k1, k2, genotype)

    def clear(self):
        self._model_ref = None
        self._model_alt = None

    def logpmf(self, k1, k2, genotype):
        if genotype in self._logpmf_cache:
            return self._logpmf_cache[genotype]

        ref_probs = self._model_ref.logpmf(k1, 2-genotype)
        alt_probs = self._model_alt.logpmf(k2, genotype)
        prob = ref_probs + alt_probs

        self._logpmf_cache[genotype] = prob
        return prob


class ChunkedComboModelBothAlleles(Model):
    """Similar as ComboModelBothAlleles, but consists of
    multiple ComboModelBothAlleles (for chunks of variants).
    Assumes logpmf is cached for all the models (logpmfs already computed)
    """
    def __init__(self, models):
        # create a logpmf array with the logpmfs from each model
        self._logpmf = {
            genotype: np.concatenate([model.logpmf(None, None, genotype) for model in models])
            for genotype in [0, 1, 2]
        }

    def logpmf(self, k1, k2, genotype):
        return self._logpmf[genotype]


class HelperModel(Model):
    def __init__(self, model, helper_variants, genotype_combo_matrix, tricky_variants=None):
        self._model = model
        self._helper_variants = helper_variants
        t = time.perf_counter()
        self._genotype_probs = np.log(genotype_combo_matrix/genotype_combo_matrix.sum(axis=(-1, -2), keepdims=True))
        logging.info("Computing genotype probs in HelperModel init took %.4f sec" % (time.perf_counter()-t))
        self._tricky_variants = tricky_variants
        self.count_probs = None

    def score(self, k1, k2):
        count_probs = np.array([self._model.logpmf(k1, k2, g) for g in [0, 1, 2]]).T
        self.count_probs = count_probs

        if self._tricky_variants is not None:
            logging.info("Using tricky variants in HelperModel.score. There are %d tricky variants" % np.sum(self._tricky_variants))
            count_probs = np.where(self._tricky_variants.reshape(-1, 1), np.log(1/3), count_probs)

        time_start = time.perf_counter()
        log_probs =  self._genotype_probs + count_probs[self._helper_variants].reshape(-1, 3, 1)+count_probs.reshape(-1, 1, 3)
        logging.debug("Time spent on log_probs in HelperModel.score: %.4f" % (time.perf_counter()-time_start))
        time_start = time.perf_counter()
        #result = logsumexp(log_probs, axis=H)
        #result = result - logsumexp(result, axis=-1, keepdims=True)
        result = run_numpy_based_function_in_parallel(lambda p: logsumexp(p, axis=H), 16, [log_probs])
        result = run_numpy_based_function_in_parallel(lambda result: result - logsumexp(result, axis=-1, keepdims=True), 16, [result])
        logging.debug("Time spent to compute probs using helper probs in HelperModel.score: %.4f" % (time.perf_counter()-time_start))
        return result

    def logpmf(self, ref_counts, alt_counts, genotype):
        count_probs = np.array([self._model.logpmf(ref_counts, alt_counts, g) for g in [0, 1, 2]]).T
        log_probs =  self._genotype_probs+count_probs[self._helper_variants].reshape(-1, 3, 1)+count_probs.reshape(-1, 1, 3)
        return logsumexp(log_probs, axis=H)[..., genotype]
