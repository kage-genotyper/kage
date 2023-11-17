from kage.models.models import *
import numpy as np
from kage.indexing.div import get_filter_of_variants_part_of_multiallelic_variant

class DummyBoth:

    def __init__(self, prob):
        self._prob = np.asanyarray(prob)

    def logpmf(self, k1, k2, genotype, base_lambda=1, gpu=False, n_threads=1):
        return np.log(self._prob[:, genotype])


pi = np.array([[0.4, 0.2, 0],
               [0.2, 0.1, 0],
               [0.1, 0,   0]])

true_pmfs = [[0.1*(0.4*0.8 + 0.2*0.1 + 0.1*0.1  ),
              0.8*(0.4*0.1 + 0.2*0.2 + 0.0*0.3  ),
              0.3*(0.4*0.1 + 0.2*0.2 + 0.1*0.3)],
             [0.2*(0.2*0.8 + 0.1*0.1 + 0.0*0.1  ),
              0.1*(0.2*0.1 + 0.1*0.2 + 0.0*0.3  ),
              0.4*(0.2*0.1 + 0.1*0.2 + 0.0*0.3)],
             [0.3*0,
              0.1*(0.1*0.1), 
              0.3*0]]

true_pmfs = np.array(true_pmfs)
true_scores = [[pmf/(true_pmfs[0][j]+true_pmfs[1][j] + true_pmfs[2][j]) for j, pmf in enumerate(row)]
               for row in true_pmfs]

# true_pmfs/true_pmfs.sum(axis=0, keepdims=True)

dummy = DummyBoth([[0.1, 0.2, 0.3],
                       [0.8, 0.1, 0.1],
                       [0.3, 0.4, 0.3]])
combo_matrix = np.array([pi, pi.T, pi*2])
helper_model = HelperModel(dummy, [1, 0, 0], combo_matrix)


def test_helper_model_pmf():
    pmf = np.exp(helper_model.logpmf(None, None, 0))
    assert np.allclose(pmf, true_pmfs[0], atol=0.02)
    pmf_1 = np.exp(helper_model.logpmf(None, None, 1))
    assert np.allclose(pmf_1, true_pmfs[1], atol=0.02)

    pmf_2 = np.exp(helper_model.logpmf(None, None, 2))
    assert np.allclose(pmf_2, true_pmfs[2], atol=0.02)


def test_helper_model_scores():
    scores = np.exp(helper_model.score(None, None))
    assert np.allclose(scores.T, true_scores, atol=0.02)


class VariantMock:
    ref_nodes = np.array([None, 0, 2, 4, None])
    var_nodes = np.array([None, 1, 3, 5, None])

class CountMock:
    def get_node_count_array():
        return 

class Mockup:
    _variant_to_nodes = VariantMock
    _min_variant_id = 1
    _max_variant_id = 3
    

def test_get_filter_of_variants_part_of_multiallelic():
    n_alleles = np.array([2, 2, 4, 2, 5])
    filter = get_filter_of_variants_part_of_multiallelic_variant(n_alleles)
    correct = [False, False, True, True, True, False, True, True, True, True]

    assert np.all(filter == correct)

