import numpy as np
from scipy.stats import poisson
from npstructures import RaggedArray
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


@dataclass
class RaggedFrequencySamplingComboModel:
    having_nodes_kmer_haplo_tuple: (RaggedArray, RaggedArray)
    not_having_nodes_kmer_haplo_tuple: (RaggedArray, RaggedArray)

    @classmethod
    def _get_kmer_haplo_tuple(cls, counts_ragged_array, base_lambda):
        kmer_counts = []
        haplo_counts = []
        # kmer_counts, haplo_counts = zip(*(np.unique(row, return_counts=True) for row in counts_ragged_array))
        for row in counts_ragged_array:
            unique, counts = np.unique(row, return_counts=True)
            kmer_counts.append(unique)
            haplo_counts.append(counts)

        return (RaggedArray(kmer_counts)*base_lambda, RaggedArray(haplo_counts))

    @classmethod
    def from_counts(cls, base_lambda, counts_having_nodes, counts_not_having_nodes):
        return cls(cls._get_kmer_haplo_tuple(counts_having_nodes, base_lambda),
                   cls._get_kmer_haplo_tuple(counts_not_having_nodes, base_lambda))

    def __eq__(self, other):
        t = True
        for i in range(2):
            t &= np.all(self.having_nodes_kmer_haplo_tuple[i]==other.having_nodes_kmer_haplo_tuple[i])
            t &= np.all(self.not_having_nodes_kmer_haplo_tuple[i]==other.not_having_nodes_kmer_haplo_tuple[i])
        return t
