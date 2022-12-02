import pytest
import numpy as np

from kage.models.sampling_combo_model import LimitedFrequencySamplingComboModel
from scipy.stats import poisson

n_haplotypes = 4



@pytest.fixture
def observed_counts():
    return np.array([1, 0])




@pytest.fixture
def matrix_combomodel():
    return LimitedFrequencySamplingComboModel.create_naive(n_variants=2, max_count=4)


def test_matrix_combomodel_logpmf(matrix_combomodel, observed_counts, diplotype=2):
    scores = matrix_combomodel.logpmf(observed_counts, diplotype)
    scores = np.exp(scores)

    truth = []
    for node in range(len(observed_counts)):
        prob = 0
        for possible_count, n_individuals in enumerate(matrix_combomodel.diplotype_counts[diplotype][node]):
            ratio = n_individuals / np.sum(matrix_combomodel.diplotype_counts[diplotype][node])
            prob += ratio * poisson.pmf(observed_counts[node], possible_count)
            #print("Possible count: %d, n_individuals: %.2f, ratio: %.2f" % (possible_count, n_individuals, ratio))
        truth.append(prob)

    print(truth)
    print(scores)

    assert np.allclose(truth, scores, atol=0.01)


def test_fill_matrix_empty_data():
    model = LimitedFrequencySamplingComboModel([
        np.array([[3, 0, 0, 0],
                 [2, 0, 0, 10]]),
        np.array([[0, 4, 0, 0],
                 [0, 0, 5, 0]]),
        np.array([[0, 0, 10, 0],
                  [0, 0, 20, 0]])
    ])

    model.astype(float)
    model.fill_empty_data(prior=0.1)
