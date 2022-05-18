import pytest
import numpy as np
from npstructures import RaggedArray

from kage.sampling_combo_model import SimpleSamplingComboModel, RaggedFrequencySamplingComboModel, LimitedFrequencySamplingComboModel
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
        counts_having_nodes, counts_not_having_nodes) == simple_sampling_combo_model


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
        [counts_having_0, counts_having_1, counts_having_2]) == ragged_frequency_sampling_combo_model


@pytest.fixture
def ragged_frequency_sampling_model_with_missing_data():
    return RaggedFrequencySamplingComboModel(
        [
            (RaggedArray([[0], [0, 1]]), RaggedArray([[5], [2, 1]])),
            (RaggedArray([[1], [1, 2]]), RaggedArray([[5], [2, 3]])),
            (RaggedArray([[], [1, 2]]), RaggedArray([[], [2, 3]]))
        ]
    )

@pytest.fixture
def ragged_frequency_sampling_model_with_all_missing_data():
    return RaggedFrequencySamplingComboModel(
        [
            (RaggedArray([[], [0, 1]]), RaggedArray([[], [2, 1]])),
            (RaggedArray([[], [1, 2]]), RaggedArray([[], [2, 3]])),
            (RaggedArray([[], [1, 2]]), RaggedArray([[], [2, 3]]))
        ]
    )

@pytest.fixture
def ragged_frequency_sampling_model_with_only_0_having_data():
    return RaggedFrequencySamplingComboModel(
        [
            (RaggedArray([[0], [0, 1]]), RaggedArray([[2], [2, 1]])),
            (RaggedArray([[], [4, 2]]), RaggedArray([[], [2, 2]])),
            (RaggedArray([[], [1, 2]]), RaggedArray([[], [2, 3]]))
        ]
    )


def test_fill_missing_data(ragged_frequency_sampling_model_with_missing_data):
    m = ragged_frequency_sampling_model_with_missing_data
    new = m.fill_empty_data()
    assert np.all(new.diplotype_counts[2][0][0] == [2])
    assert np.all(new.diplotype_counts[2][1][0] == [1])


def test_fill_missing_data_should_crash(ragged_frequency_sampling_model_with_all_missing_data):
    m = ragged_frequency_sampling_model_with_all_missing_data
    with pytest.raises(AssertionError):
        new = m.fill_empty_data()

def test_fill_missing_data_only_0_having_data(ragged_frequency_sampling_model_with_only_0_having_data):
    m = ragged_frequency_sampling_model_with_only_0_having_data
    new = m.fill_empty_data()
    assert np.all(new.diplotype_counts[1][0][0] == [3])  # 3 is average of other data for 1
    assert np.all(new.diplotype_counts[1][1][0] == [1])  # frequency is 1 for missing data


# @pytest.mark.skip("waiting")
@pytest.mark.parametrize("diplotype", [0])#, 1, 2])
def test_combomodel_logpmf(ragged_frequency_sampling_combo_model, observed_counts, diplotype):
    model = ragged_frequency_sampling_combo_model
    scores = model.logpmf(observed_counts, diplotype)
    n_diplotypes_for_having = np.array([[1, 3],  # 0
                                        [3, 1],  # 1
                                        [3, 4]]) # 2
    kmer_counts, diplo_counts = model.diplotype_counts[diplotype]
    truth = []
    for node in range(2):
        truth.append(
            sum(diplo_counts[node]/n_diplotypes_for_having[diplotype, node]*np.exp(
                poisson.logpmf(observed_counts[node], kmer_counts[node]))))
    assert np.allclose(scores, np.log(truth))
    # 
    # 
    # for node, count in enumerate(observed_counts):
    #     logpmf_having = sum(model.having_nodes_kmer_haplo_tuple[1]/n_haplotypes_having[node]*poisson.logpmf(count, model.having_nodes_kmer_haplo_tuple[0]))
    #     logpmf_not_having = sum(model.not_having_nodes_kmer_haplo_tuple[1]/n_haplotypes_not_having[node]*poisson.logpmf(count, model.not_having_nodes_kmer_haplo_tuple[0]))
    # 
    # 
    # truth = [[poisson.logpmf(oc, e) for e in expected]
    #          for oc, expected in
    #          zip(observed_counts, simple_sampling_combo_model.expected)]
    # 
    # assert np.all(score == truth)


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

    assert np.all(truth == scores)

