from kage.util import n_unique_values_per_column
import numpy as np


def test_n_unique_values_per_column():
    a = np.array([
        [1, 5, 3, 1],
        [1, 3, 3, 10],
        [1, 5, 4, 1],
        [1, 0, 4, 5]
    ])

    assert np.all(n_unique_values_per_column(a) == [1, 3, 2, 3])
