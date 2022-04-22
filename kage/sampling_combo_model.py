import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp
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
    diplotype_counts: list((RaggedArray, RaggedArray))
    # having_nodes_kmer_haplo_tuple: (RaggedArray, RaggedArray)
    # not_having_nodes_kmer_haplo_tuple: (RaggedArray, RaggedArray)

    @classmethod
    def _get_kmer_diplo_tuple(cls, counts_ragged_array, base_lambda):
        kmer_counts = []
        haplo_counts = []
        # kmer_counts, haplo_counts = zip(*(np.unique(row, return_counts=True) for row in counts_ragged_array))
        for row in counts_ragged_array:
            unique, counts = np.unique(row, return_counts=True)
            kmer_counts.append(unique)
            haplo_counts.append(counts)

        return (RaggedArray(kmer_counts)*base_lambda, RaggedArray(haplo_counts))

    @classmethod
    def from_counts(cls, base_lambda, diplotype_counts):

        return cls([cls._get_kmer_diplo_tuple(counts, base_lambda) for counts in diplotype_counts])

    def logpmf(self, observed_counts, d):
        kmer_counts, diplo_counts = self.diplotype_counts[d]
        diplo_counts = diplo_counts.astype(float)
        kmer_counts = kmer_counts.astype(float)
        s = diplo_counts.sum(axis=-1)
        p_diplo_count = np.log(diplo_counts/s[:, None])
        broadcasted_counts = kmer_counts.shape.broadcast_values(observed_counts[:, None])
        p_kmer_counts = RaggedArray(poisson.logpmf(broadcasted_counts, kmer_counts.ravel()), kmer_counts.shape)
        ps = p_kmer_counts+p_diplo_count
        return np.array([logsumexp(row) for row in ps])
        # sum(diplo_counts[node]/n_diplotypes_for_having[diplotype, node]*np.exp(poisson.logpmf(observed_counts[node], kmer_counts[node]))))


    def __eq__(self, other):
        return all(np.all(s[0]==o[0]) and np.all(s[1] == o[1]) for s, o in zip(self.diplotype_counts, other.diplotype_counts))
