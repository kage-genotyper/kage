import logging
import time
import scipy
import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp
from npstructures import RaggedArray
from dataclasses import dataclass
from typing import List
from .util import log_memory_usage_now
from .models import Model
from shared_memory_wrapper.shared_memory import run_numpy_based_function_in_parallel


def fast_poisson_logpmf(k, r):
    # should return the same as scipy.stats.logpmf(k, r), but is  ~4x faster
    #return k  * np.log(r) - r - np.log(scipy.special.factorial(k))
    return k  * np.log(r) - r - scipy.special.gammaln(k+1)


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
class LimitedFrequencySamplingComboModel(Model):
    # stores only number of individuals having counts up to a given limit
    diplotype_counts: List[np.ndarray]  # list of matrices, each matrix is n_variants x max count supported

    def __post_init__(self):
        #self.diplotype_counts = [c.astype(float) for c in self.diplotype_counts]
        return

    def astype(self, dtype):
        for i in range(3):
            #logging.info(self.diplotype_counts[i])
            self.diplotype_counts[i] = self.diplotype_counts[i].astype(dtype)

    def add_error_rate(self, error_rate=0.1):
        pass

    @classmethod
    def create_empty(cls, n_variants, max_count=3):
        logging.info("Creating empty limited freq model with dimensions %d x %d "% (n_variants, max_count))
        log_memory_usage_now("Before creating empty")
        ret = cls([np.zeros((n_variants, max_count), dtype=np.uint16) for i in range(3)])
        log_memory_usage_now("After creating empty")
        return ret

    def __add__(self, other):
        for i in range(3):
            self.diplotype_counts[i] += other.diplotype_counts[i]
        return self

    def __getitem__(self, item):
        return self.diplotype_counts[item]

    @classmethod
    def create_naive(cls, n_variants, max_count=3, prior=0):
        empty = np.zeros((n_variants, max_count))
        counts0 = empty.copy()
        counts0[:,0] = 1
        counts0[:,0] += prior
        counts1 = empty.copy()
        counts1[:,1] = 1
        counts1[:,1] += prior
        counts2 = empty.copy()
        counts2[:,2] = 1
        counts2[:,2] += prior
        return cls([counts0, counts1, counts2])

    def subset_on_nodes(self, nodes):
        return self.__class__([matrix[nodes,:] for matrix in self.diplotype_counts])

    def __eq__(self, other):
        return all(np.all(m1 == m2) for m1, m2 in zip(self.diplotype_counts, other.diplotype_counts))

    @staticmethod
    def _logpmf(observed_counts, counts, base_lambda, error_rate):
        sums = np.sum(counts, axis=-1)[:,None]
        frequencies = np.log(counts / sums)
        poisson_lambda = (np.arange(counts.shape[1])[None,:] + error_rate) * base_lambda
        prob = fast_poisson_logpmf(observed_counts[:,None], poisson_lambda)
        prob = logsumexp(frequencies + prob, axis=-1)
        return prob


    def logpmf(self, observed_counts, d, base_lambda=1.0, error_rate=0.01):
        logging.debug("base lambda in LimitedFreq model is %.3f" % base_lambda)
        logging.debug("Error rate is %.3f" % error_rate)
        t0 = time.perf_counter()
        counts = self.diplotype_counts[d]
        counts = counts.astype(np.float16)
        prob = run_numpy_based_function_in_parallel(
            LimitedFrequencySamplingComboModel._logpmf, 16, (observed_counts, counts, base_lambda, error_rate)
        )
        logging.debug("Logpmf took %.4f sec" % (time.perf_counter()-t0))

        return prob
        #return LimitedFrequencySamplingComboModel._logpmf(observed_counts, counts, base_lambda, error_rate)



        t0 = time.perf_counter()
        sums = np.sum(counts, axis=-1)[:,None]
        logging.info("Sums took %.4f sec" % (time.perf_counter()-t0))

        #frequencies = np.log(counts / sums)
        t0 = time.perf_counter()
        frequencies = run_numpy_based_function_in_parallel(np.log, 16, (counts/sums,))
        logging.debug("Frequencies took %.4f sec" % (time.perf_counter()-t0))

        poisson_lambda = (np.arange(counts.shape[1])[None,:] + error_rate) * base_lambda

        #prob = poisson.logpmf(observed_counts[:,None], poisson_lambda)
        t0 = time.perf_counter()
        prob = run_numpy_based_function_in_parallel(fast_poisson_logpmf, 16, (observed_counts[:,None], poisson_lambda))
        logging.debug("Prob took %.4f sec" % (time.perf_counter()-t0))


        #prob = logsumexp(frequencies + prob, axis=-1)
        t0 = time.perf_counter()
        prob = run_numpy_based_function_in_parallel(lambda x: logsumexp(x, axis=-1), 16, (frequencies+prob,))
        logging.debug("Prob2 took %.4f sec" % (time.perf_counter()-t0))

        logging.debug("LimitedFreqModel logpmf took %.4f sec" % (time.perf_counter()-t))
        return prob
        # sum(diplo_counts[node]/n_diplotypes_for_having[diplotype, node]*np.exp(poisson.logpmf(observed_counts[node], kmer_counts[node]))))

    def describe_node(self, node):
        description = "\n"
        for count in range(3):
            description += "Having %d copies: " % count
            description += ', '.join("%d: %.3f" % (i, self.diplotype_counts[count][node][i]) for i in np.nonzero(self.diplotype_counts[count][node])[0])
            #description += ', '.join("%d: %d" % (c, f) for c, f in np.unique(self.diplotype_counts[count][node], return_counts=True))
            description += "\n"

        return description

    def fill_empty_data(self, prior=0.1):
        t = time.perf_counter()
        logging.info("Prior is %.4f" % prior)
        expected_counts = []
        for diplotype in [0, 1, 2]:
            logging.info("Computing expected counts for genotype %d" % diplotype)
            m = self.diplotype_counts[diplotype]
            not_missing = np.where(np.sum(m, axis=-1) > 0)[0]
            expected = np.zeros(m.shape[0], dtype=float)
            expected[not_missing] = np.sum(np.arange(m.shape[1]) * m[not_missing], axis=-1) / np.sum(m[not_missing], axis=-1)
            expected_counts.append(expected)

        # add priors
        # assume naively counts for 1 is 1 more than expected as 0. Counts at 2 is 2 more than expected at 1
        e0, e1, e2 = expected_counts
        m0, m1, m2 = self.diplotype_counts
        max_count = m0.shape[1]-1
        logging.debug("Adding priors")
        logging.debug("Size of e: %s" % (e1.shape))
        positions = np.round(e1).astype(int) - 1
        logging.debug("Found positions")
        positions = np.maximum(0, positions)
        logging.debug("Found max")
        rows = np.arange(0, m0.shape[0])
        m0[rows, positions] += prior
        m1[rows,np.minimum(max_count, np.round(e0).astype(int) + 1)] += prior
        m2[rows,np.minimum(max_count, np.round(e0).astype(int) + 2)] += prior

        assert all([np.all(np.sum(c, axis=-1) > 0) for c in self.diplotype_counts])
        logging.debug("Filling empty data took %.2f sec " % (time.perf_counter()-t))

    def has_no_data(self, idx):
        missing = [np.sum(c[idx]) == 0 for c in self.diplotype_counts]
        # if 2 out of 3 is missing data, return True
        return sum(missing) >= 2

    def has_duplicates(self, idx):
        counts = [c[idx] for c in self.diplotype_counts]
        for i in range(3):
            # if there are counts outside position i, there are duplicates
            # also if any elements outside i, there are duplicates
            #if np.sum(counts[i]) > counts[i][i] * 5:
            if np.sum(counts[i][i+1:]) > 0:
            #if np.argmax(counts[i]) != i:
                return True

        return False


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
        kmer_counts, haplo_counts = np.unique(counts_ragged_array, axis=-1, return_counts=True)
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
        total_counts = [(counts * frequencies) for counts, frequencies in self.diplotype_counts]
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

    @classmethod
    def from_multiple(cls, models):
        # merges multiple models into a single
        new = []
        for count in range(3):
            new_counts = []
            new_frequencies = []



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
