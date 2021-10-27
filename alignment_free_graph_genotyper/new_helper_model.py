from scipy.special import logsumexp
import numpy as np

MAIN = -1
HELPER = -2
M = MAIN
H = HELPER

def create_combined_matrices(genotype_matrix, window_size):
    helper = genotype_matrix*3
    result = []
    for i in range (1, window_size):
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

class HelperModel:
    def __init__(self, model, helper_variants, genotype_combo_matrix):
        self._model = model
        self._helper_variants = helper_variants
        self._genotype_probs = np.log(genotype_combo_matrix/genotype_combo_matrix.sum(axis=-1, keepdims=True))

    def predict(self, k1, k2):
        probs = [self.logpmf(k1, k2, g) for g in range(3)]
        return np.argmax(probs, axis=0)

    @classmethod
    def from_genotype_matrix(cls, model, genotype_matrix):
        combined = create_combined_matrices(genotype_matrix, 10)
        helpers = find_best_helper(combined, calc_likelihood)
        # helpers = find_best_helper(combined, calc_argmax)
        helper_counts = genotype_matrix[helpers]*3
        flat_idx = genotype_matrix+helper_counts
        genotype_combo_matrix = np.array([(flat_idx==k).sum(axis=1) for k in range(9)]).T.reshape(-1, 3, 3)+1
        return cls(model, helpers, genotype_combo_matrix)

    def logpmf(self, ref_counts, alt_counts, genotype):
        count_probs = np.array([self._model.logpmf(ref_counts, alt_counts, g) for g in [0, 1, 2]]).T
        log_probs =  self._genotype_probs+count_probs[self._helper_variants].reshape(-1, 3, 1)+count_probs.reshape(-1, 1, 3)
        return logsumexp(log_probs, axis=H)[..., genotype]

class SimpleHelperModel(HelperModel):
    def logpmf(self, ref_counts, alt_counts, genotype):
        count_probs = np.array([self._model.logpmf(ref_counts, alt_counts, g) for g in [0, 1, 2]]).T
        helper_g = np.argmax(count_probs[self._helper_variants]+logsumexp(self._genotype_probs[self._helper_variants], axis=M), axis=-1)
        res =  count_probs[:, genotype]+np.array([self._genotype_probs[i, h, genotype] for i, h in  enumerate(helper_g)])# self._genotype_probs[:, helper_g, genotype]
        assert res.shape == (self._genotype_probs.shape[0],), res.shape
        return res

class PriorModel:
    def __init__(self, model, genotype_counts):
        self._model = model
        print(genotype_counts)
        self._genotype_frequencies = np.log(genotype_counts/genotype_counts.sum(axis=-1, keepdims=True))
        assert np.allclose(logsumexp(self._genotype_frequencies, axis=-1), 0)

    def predict(self, k1, k2):
        probs = [self.logpmf(k1, k2, g) for g in range(3)]
        return np.argmax(probs, axis=0)

    @classmethod
    def from_genotype_matrix(cls, model, genotype_matrix):
        genotype_counts = np.array([np.sum(genotype_matrix==k, axis=-1) for k in range(3)]).T
        return cls(model, genotype_counts)

    def logpmf(self, ref_counts, alt_counts, genotype):
        return self._model.logpmf(ref_counts, alt_counts, genotype)+self._genotype_frequencies[:, genotype]


"""
sum_(g_H) P(G_H)*P(g_H|G_H)*
"""
def calc_likelihood_for(count_matrix, prob_dists):
    """
    count_matrix = [g_h,g_m]
    p[g_h,k] = prob of observing k of total_reads on helper ref if gneotype is g_h on helper variant
    """
    N = prob_dists.shape[-1]-1
    t = 0 
    M  = count_matrix
    p = prob_dists
    for g_h in range(3):
        for g_m in range(3):
            for k in range(N+1):
                t += M[g_h, g_m]*p[g_h, k]*np.log(sum([p[x, k]*M[x, g_m]/np.sum(p[:, k]) for x in range(3)]))
    return t

def get_scores(count_matrices, prob_dists):
    return np.array([calc_likelihood_for(count_matrix, prob_dist)
                     for count_matrix, prob_dist in zip(count_matrices, prob_dists)])
        

def full_solution(combined, prob_dists):
    """
    combined: (w, n-1->n-w, 3, 3)
    prob_dists: (n, 3, total_reads)
    p[v,g,k] = prob of observing k of total_reads on ref if gneotype ig on varaint v
    """
    N = len(combined[0])+1
    best_idx, best_score = np.empty(N), -np.inf*np.ones(N)
    for j, counts in enumerate(combined, 1):
        
        scores = get_scores(counts, prob_dists[:-j])
        do_update = scores>best_score[j:]
        best_score[j:][do_update] = scores[do_update]
        best_idx[j:][do_update] = np.flatnonzero(do_update)
        rev_scores = get_scores(counts.swapaxes(-2, -1), prob_dists[j:])
        do_update = rev_scores>best_score[:-j]
        best_score[:-j][do_update] = rev_scores[do_update]
        best_idx[:-j][do_update] = np.flatnonzero(do_update)+j
    return best_idx

def simulate_prob_dists(n_variants, N):
    ps = np.random.rand(n_variants, 3, N)
    return ps/ps.sum(axis=-1, keepdims=True)
