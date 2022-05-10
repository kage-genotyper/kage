import logging
import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp
from npstructures import RaggedArray
from dataclasses import dataclass
from typing import List


@dataclass
class SimpleSamplingComboModel:
    expected: np.ndarray # N_nodes x 3

    @classmethod
    def from_counts(cls, counts_having_nodes, counts_not_having_nodes):
        means_with_node = counts_having_nodes.mean(axis=-1)
        means_without_node = counts_not_having_nodes.mean(axis=-1)
        expected_0 = 2*means_without_node
        expected_1 = means_without_node+means_with_node
        expected_2 = 2*means_with_node
        return cls(np.array([expected_0, expected_1, expected_2]).T)

    def logpmf(self, observed_counts):
        return poisson.logpmf(observed_counts[:, None], self.expected)

    def __eq__(self, other):
        return np.all(self.expected == other.expected)

@dataclass
class RaggedFrequencySamplingComboModel:
    # the first RaggedArrray in the tuple represents counts,
    # the second frequencies (how many have that count)
    diplotype_counts: List[List[RaggedArray]]

    def add_error_rate(self, error_rate=0.1):
        for i in range(3):
            self.diplotype_counts[i][0]._data += error_rate

    def astype(self, type):
        for i in range(3):
            self.diplotype_counts[i][0] = self.diplotype_counts[i][0].astype(type)

    def scale(self, base_lambda):
        for i in range(3):
            self.diplotype_counts[i][0]._data *= int(base_lambda)

    @classmethod
    def create_naive(cls, n):
        return cls([
            [RaggedArray([0] * n, [1] * n), RaggedArray([1]*n, [1]*n)],
            [RaggedArray([1] * n, [1] * n), RaggedArray([1]*n, [1]*n)],
            [RaggedArray([2] * n, [1] * n), RaggedArray([1]*n, [1]*n)]
        ])

    @classmethod
    def _get_kmer_diplo_tuple(cls, counts_ragged_array):
        kmer_counts = []
        haplo_counts = []
        # kmer_counts, haplo_counts = zip(*(np.unique(row, return_counts=True) for row in counts_ragged_array))
        kmer_counts, haplo_counts = np.unique(counts_ragged_array, axis=-1, return_counts=True)

        """
        for i, row in enumerate(counts_ragged_array):
            if i % 1000 == 0:
                logging.info("Making diplo counts for variant %d/%d" % (i, len(counts_ragged_array)))
            unique, counts = np.unique(row, return_counts=True)
            kmer_counts.append(unique)
            haplo_counts.append(counts)
        """

        return [RaggedArray(kmer_counts), RaggedArray(haplo_counts)]

    def describe_node(self, node):
        description = "\n"
        for count in range(3):
            description += "Having %d copies: " % count
            description += ', '.join("%d: %d" % (c, f) for c, f in zip(
                self.diplotype_counts[count][0][node], self.diplotype_counts[count][1][node]
            ))
            description += "\n"

        return description

    def fill_empty_data(self):
        logging.info("Filling empty data")
        # fill in estimates where there is missing data
        # assuming all entries are only missing at most one of the three
        # if missing at 2:
        #     assume diff between exp count at 1 minus 0 added to the count for 1
        #     c2 = c1 + c1 - c0
        # if missing at 1:
        #   c1 = (c2 + c0) / 2
        # if missing at 0
        #   c0 = 2*c1 - c2

        # first compute expected counts
        #sums = [np.array([np.sum(row) if len(row) > 0 else 1 for row in frequencies])
        #        for _, frequencies in self.diplotype_counts]
        sums = [np.sum(frequencies, axis=-1) for _, frequencies in self.diplotype_counts]
        print()
        print("SUMS: %s" % sums)
        logging.info("N sums: %d" % len(sums[0]))

        total_counts = [(counts * frequencies) for counts, frequencies in self.diplotype_counts]
        logging.info("N total counts: %d" % len(total_counts[0]))
        #expected = [np.array([np.sum(row) if len(row) > 0 else -1 for row in rows]) for rows in total_counts]
        expected = [np.sum(s, axis=-1) for s in total_counts]  # alternative
        expected = [e/s for e, s in zip(expected, sums)]

        c0, c1, c2 = expected
        missing0, missing1, missing2 = [counts.shape.lengths == 0 for counts, _ in self.diplotype_counts]

        assert np.sum(missing1 & missing2 & missing0) == 0, "Cannot have mising data for all genotypes"

        # if both 0 and 2 are missing, we set 2 to double of 1
        c2[missing0 & missing2] = 2 * c1[missing0 & missing2]
        missing2[missing0 & missing2] = False

        # if both 0 and 1 are missing, set 1 to half of 2
        c1[missing0 & missing1] = c2[missing0 & missing1] / 2
        missing1[missing0 & missing1] = False

        # if both 1 and 2 are missing, we guess that 1 will have the average of all other 1's
        mean_of_c1 = np.mean(c1[~missing1])
        c1[missing1 & missing2] = np.mean(c1[~missing1])
        missing1[missing1 & missing2] = False  # 1 is now not missing at these positions anymore

        # compute new expected where only one is missing
        c0[missing0] = 2 * c1[missing0] - c2[missing0]
        c1[missing1] = (c2[missing1] + c0[missing1]) / 2
        c2[missing2] = c1[missing2] + c1[missing2] - c0[missing2]

        w = np.where(np.isnan(c0[missing0]))
        print(w)
        print(c1[w])
        print(c2[w])

        assert np.sum(np.isnan(c0[missing0])) == 0
        assert np.sum(np.isnan(c1[missing1])) == 0
        assert np.sum(np.isnan(c2[missing2])) == 0

        # create new ragged arrays
        # set count to expected and frequency to 1 where data is missing
        all_expected = [c0, c1, c2]
        #all_missing = [missing0, missing1, missing2]
        new_diplotype_counts = []
        for expected, (old_counts, old_frequencies) in zip(all_expected, self.diplotype_counts):
            new_frequencies = RaggedArray([
                old if len(old) > 0 else [1] for old in old_frequencies
            ])
            new_counts = RaggedArray([
                old if len(old) > 0 else [int(e)] for old, e in zip(old_counts, expected)
            ])
            new_diplotype_counts.append([new_counts, new_frequencies])

        return RaggedFrequencySamplingComboModel(new_diplotype_counts)


    @classmethod
    def from_counts(cls, diplotype_counts):
        return cls([cls._get_kmer_diplo_tuple(counts) for counts in diplotype_counts])

    def logpmf(self, observed_counts, d, error_rate=0.0):
        kmer_counts, diplo_counts = self.diplotype_counts[d]
        diplo_counts = diplo_counts.astype(float)
        kmer_counts = kmer_counts.astype(float) + error_rate
        s = diplo_counts.sum(axis=-1)
        p_diplo_count = np.log(diplo_counts/s[:, None])
        broadcasted_counts = kmer_counts.shape.broadcast_values(observed_counts[:, None])
        p_kmer_counts = RaggedArray(poisson.logpmf(broadcasted_counts, kmer_counts.ravel()), kmer_counts.shape)
        ps = p_kmer_counts+p_diplo_count
        return np.array([logsumexp(row) for row in ps])
        # sum(diplo_counts[node]/n_diplotypes_for_having[diplotype, node]*np.exp(poisson.logpmf(observed_counts[node], kmer_counts[node]))))

    def subset_on_nodes(self, nodes):
        return RaggedFrequencySamplingComboModel([[counts[nodes], frequencies[nodes]] for counts, frequencies in self.diplotype_counts])

    def __eq__(self, other):
        return all(np.all(s[0]==o[0]) and np.all(s[1] == o[1]) for s, o in zip(self.diplotype_counts, other.diplotype_counts))
