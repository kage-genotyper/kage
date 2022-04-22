import numpy as np
from scipy.stats import poisson

from dataclasses import dataclass


@dataclass
class SimpleSamplingComboModel:
    expected: np.ndarray # N_nodes x 3

    @classmethod
    def from_counts(cls, base_lambda, counts_having_nodes, counts_not_having_nodes):
        means_with_node = counts_having_nodes.mean(axis=-1)
        means_without_node = counts_not_having_nodes.mean(axis=-1)
        expected_0 = 2*means_without_node
        expected_1 = means_without_node+means_with_node
        expected_2 = 2*means_with_node
        return cls(np.array([expected_0, expected_1, expected_2]).T*base_lambda)

    def logpmf(self, observed_counts):
        return poisson.logpmf(observed_counts[:, None], self.expected)

    def __eq__(self, other):
        return np.all(self.expected == other.expected)
