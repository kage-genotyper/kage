import pytest
import numpy as np
from npstructures import RaggedArray

from kage.sampling_combo_model import SimpleSamplingComboModel, RaggedFrequencySamplingComboModel
from scipy.stats import poisson

n_haplotypes = 4


@pytest.fixture
def counts_having_nodes():
    return RaggedArray([[1, 1, 2],
                        [3]])

@pytest.fixture
def counts_not_having_nodes():
    return RaggedArray([[0],
                        [0, 0, 1]])

@pytest.fixture
def observed_counts():
    return np.array([1, 0])


@pytest.fixture
def simple_sampling_combo_model():
    return SimpleSamplingComboModel(
        np.array([[0, 4/3, 8/3],
                  [2/3, 1/3+3, 3*2]]))


@pytest.fixture
def ragged_frequency_sampling_combo_model():
    return RaggedFrequencySamplingComboModel(
        (RaggedArray([[1, 2], [3]]), RaggedArray([[2, 1], [1]])),
        (RaggedArray([[0], [0, 1]]), RaggedArray([[1], [2, 1]])))


def test_simple_from_counts(counts_having_nodes, counts_not_having_nodes, 
                            simple_sampling_combo_model):
    assert SimpleSamplingComboModel.from_counts(
        1, counts_having_nodes, counts_not_having_nodes) == simple_sampling_combo_model


# @pytest.mark.skip("unimplenented")
def test_combomodel_logpmf(simple_sampling_combo_model, observed_counts):
    score = simple_sampling_combo_model.logpmf(observed_counts)
    truth = [[poisson.logpmf(oc, e) for e in expected]
             for oc, expected in
             zip(observed_counts, simple_sampling_combo_model.expected)]

    assert np.all(score == truth)


def test_frequency_from_counts(counts_having_nodes, counts_not_having_nodes,
                               ragged_frequency_sampling_combo_model):

    assert RaggedFrequencySamplingComboModel.from_counts(
        1, counts_having_nodes, counts_not_having_nodes) == ragged_frequency_sampling_combo_model
