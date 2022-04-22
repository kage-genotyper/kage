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
def counts_having_0():
    return RaggedArray([[0],
                        [0, 0, 1]])

@pytest.fixture
def counts_having_1():
    return RaggedArray([[1, 1, 2],
                        [3]])


@pytest.fixture
def counts_having_2():
    return RaggedArray([[3, 2, 1],
                        [4, 5, 6, 6]])

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
        [(RaggedArray([[0], [0, 1]]), RaggedArray([[1], [2, 1]])),
         (RaggedArray([[1, 2], [3]]), RaggedArray([[2, 1], [1]])),
         (RaggedArray([[1, 2, 3], [4, 5, 6]]), RaggedArray([[1, 1, 1], [1, 1, 2]]))])

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



def test_frequency_from_counts(counts_having_0, counts_having_1, counts_having_2,
                               ragged_frequency_sampling_combo_model):

    assert RaggedFrequencySamplingComboModel.from_counts(
        1, [counts_having_0, counts_having_1, counts_having_2]) == ragged_frequency_sampling_combo_model


@pytest.mark.skip("waiting")
def test_combomodel_logpmf(ragged_frequency_sampling_combo_model, observed_counts):
    model = ragged_frequency_sampling_combo_model
    score = model.logpmf(observed_counts)
    n_haplotypes_having = [3, 1]
    n_haplotypes_not_having = [1, 3]
    for node, count in enumerate(observed_counts):

        logpmf_having = sum(model.having_nodes_kmer_haplo_tuple[1]/n_haplotypes_having[node]*poisson.logpmf(count, model.having_nodes_kmer_haplo_tuple[0]))
        logpmf_not_having = sum(model.not_having_nodes_kmer_haplo_tuple[1]/n_haplotypes_not_having[node]*poisson.logpmf(count, model.not_having_nodes_kmer_haplo_tuple[0]))


    truth = [[poisson.logpmf(oc, e) for e in expected]
             for oc, expected in
             zip(observed_counts, simple_sampling_combo_model.expected)]

    assert np.all(score == truth)
