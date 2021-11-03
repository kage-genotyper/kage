import numpy as np
from scipy.special import logsumexp

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

    def logpmf(self, k1, k2, genotype):
        ref_probs = self._model_ref.logpmf(k1, 2-genotype)
        alt_probs = self._model_alt.logpmf(k2, genotype)
        return ref_probs+alt_probs


class HelperModel(Model):
    def __init__(self, model, helper_variants, genotype_combo_matrix, tricky_variants=None):
        self._model = model
        self._helper_variants = helper_variants
        self._genotype_probs = np.log(genotype_combo_matrix/genotype_combo_matrix.sum(axis=(-1, -2), keepdims=True))
        self._tricky_variants = tricky_variants

    def score(self, k1, k2):
        count_probs = np.array([self._model.logpmf(k1, k2, g) for g in [0, 1, 2]]).T
        log_probs =  self._genotype_probs + count_probs[self._helper_variants].reshape(-1, 3, 1)+count_probs.reshape(-1, 1, 3)
        result = logsumexp(log_probs, axis=H)
        return result - logsumexp(result, axis=-1, keepdims=True)

    def logpmf(self, ref_counts, alt_counts, genotype):
        count_probs = np.array([self._model.logpmf(ref_counts, alt_counts, g) for g in [0, 1, 2]]).T
        if self._tricky_variants is not None:
            logging.info("Using tricky variants")
            count_probs = np.where(self._tricky_variants.reshape(-1, 1), np.log(1/3), count_probs)
        log_probs =  self._genotype_probs+count_probs[self._helper_variants].reshape(-1, 3, 1)+count_probs.reshape(-1, 1, 3)
        return logsumexp(log_probs, axis=H)[..., genotype]
