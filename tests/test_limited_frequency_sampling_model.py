import numpy as np
from kage.models.models import ComboModelBothAlleles
from kage.models.sampling_combo_model import LimitedFrequencySamplingComboModel, \
    SparseLimitedFrequencySamplingComboModel, SparseObservedCounts
from numpy.testing import assert_array_equal


def test_fill_missing_data2():
    # simple case where counts for 2 are missing
    model = LimitedFrequencySamplingComboModel(
        [np.array([[0, 1, 0, 0, 0, 0]]),
         np.array([[0, 0, 1, 0, 0, 0]]),
         np.array([[0, 0, 0, 0, 0, 0]])
         ]
    )
    model.fill_empty_data2()
    assert_array_equal(model[2], [[0, 0, 0, 1, 0, 0]])

    # counts for 1 are missing
    model = LimitedFrequencySamplingComboModel(
        [np.array([[1, 1, 1, 0, 0, 0]]),
         np.array([[0, 0, 0, 0, 0, 0]]),
         np.array([[0, 0, 4, 10, 4, 0]])
         ]
    )
    model.fill_empty_data2()
    assert_array_equal(model[1], [[0, 0, 1, 0, 0, 0]])

    # counts for 0 are missing
    model = LimitedFrequencySamplingComboModel(
        [np.array([[0, 0, 0, 0, 0, 0]]),
         np.array([[0, 1, 1, 1, 0, 0]]),
         np.array([[0, 0, 4, 10, 4, 0]])
         ]
    )
    model.fill_empty_data2()
    assert_array_equal(model[0], [[0, 1, 0, 0, 0, 0]])

    # counts for two genotypes are missing (shouldn't crash)
    model = LimitedFrequencySamplingComboModel(
        [np.array([[0, 0, 0, 0, 0, 0]]),
         np.array([[0, 0, 1, 0, 0, 0]]),
         np.array([[0, 0, 0, 0, 0, 0]])
         ]
    )
    model.fill_empty_data2()

    model = LimitedFrequencySamplingComboModel(
        [np.array([[0, 0, 43, 0, 0, 0]]),
         np.array([[0, 0, 0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0, 0, 0]])
         ]
    )
    model.astype(np.float16)
    model.fill_empty_data2()
    print(model.describe_node(0))


def test_fill_missing_data_roundoff_error():
    # testing fix on roundoff error
    model = LimitedFrequencySamplingComboModel(
        [np.array([[41, 1, 0, 0, 0, 0]]),
         np.array([[0, 1, 0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0, 0, 0]])
         ]
    )
    model.fill_empty_data2()
    assert_array_equal(model[2], [[0, 0, 1, 0, 0, 0]])



def test_logpmf():
    model_ref = SparseLimitedFrequencySamplingComboModel(
        [
            SparseObservedCounts.from_nonsparse(np.array([[1, 0, 0, 0, 0]])),
            SparseObservedCounts.from_nonsparse(np.array([[0, 1, 0, 0, 0]])),
            SparseObservedCounts.from_nonsparse(np.array([[0, 0, 1, 0, 0]]))
        ]
    )
    model_alt = SparseLimitedFrequencySamplingComboModel(
        [
            SparseObservedCounts.from_nonsparse(np.array([[13, 13, 3, 0, 0]])),
            SparseObservedCounts.from_nonsparse(np.array([[0, 8, 4, 0, 0]])),
            SparseObservedCounts.from_nonsparse(np.array([[0, 0, 2, 0, 0]]))
        ]
    )
    model_both = ComboModelBothAlleles(
        model_ref, model_alt
    )

    prob = model_both.logpmf(np.array([8]), np.array([8]), 0, base_lambda=4.78125, n_threads=1)
    print("PROB\n", prob)
    prob = model_both.logpmf(np.array([8]), np.array([8]),1, base_lambda=4.78125, n_threads=1)
    print("PROB\n", prob)
