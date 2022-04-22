import pytest
import numpy as np
from npstructures import RaggedArray

from kage.sampling_combo_model import SimpleSamplingComboModel

n_haplotypes = 4


@pytest.fixture
def counts_having_nodes():
    return RaggedArray([1, 1, 2],
                       [3],)

@pytest.fixture
def counts_not_having_nodes():
    return RaggedArray([0],
                       [0, 0, 1])

@pytest.fixture
def observed_counts():
    return np.array([1, 0])


@pytest.fixture
def simple_sampling_combo_model():
    return SimpleSamplingComboModel(
        np.array([[0, 4/3, 8/3],
                  [2/3, 1/3+3, 3*2]]))


@pytest.mark.skip("unimplenented")
def test_simple_from_counts(counts_having_nodes, counts_not_having_nodes, 
                            simple_sampling_combo_model):
    assert SimpleSamplingComboModel.from_counts(
        1, counts_having_nodes, counts_not_having_nodes) == simple_sampling_combo_model


@pytest.mark.skip("unimplenented")
def test_combomodel(sampling_combo_model, observed_counts):
    score = sampling_combo_model.logpmf(observed_counts)
    assert np.all(score == None)
