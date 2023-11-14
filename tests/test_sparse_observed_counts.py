import numpy as np
from kage.models.sampling_combo_model import SparseObservedCounts, LimitedFrequencySamplingComboModel


def test():
    counts = SparseObservedCounts.from_nonsparse(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]))


    probs = counts.logpmf(np.array([0, 1.01, 2.02]), 1.0, 0.01, n_threads=1)
    print(counts)
    print(probs)


def test2():
    counts = SparseObservedCounts.from_nonsparse(
        np.zeros((3, 10))+1
    )

    probs = counts.logpmf(np.array([0, 1.01, 2.02]), 1.0, 0.01, n_threads=1)

    print(counts)
    print(probs)


def test3():
    model = LimitedFrequencySamplingComboModel(
        [
            np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 2]
            ]),
        ])

    probs = model.logpmf(np.array([1, 2, 1]), 0, 6.0, error_rate=0.001)
    print(probs)



