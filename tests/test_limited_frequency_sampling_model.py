import numpy as np
from kage.models.sampling_combo_model import LimitedFrequencySamplingComboModel
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

