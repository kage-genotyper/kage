import logging
import time
import scipy
import numpy as np
import shared_memory_wrapper.util
from scipy.stats import poisson
from scipy.special import logsumexp
from npstructures import RaggedArray
from dataclasses import dataclass
from typing import List
from kage.util import log_memory_usage_now
from kage.models.models import Model
from shared_memory_wrapper.shared_memory import run_numpy_based_function_in_parallel, object_to_shared_memory, object_from_shared_memory, remove_shared_memory
from typing import Tuple


def fast_poisson_logpmf(k, r):
    # should return the same as scipy.stats.logpmf(k, r), but is  ~4x faster
    #return k  * np.log(r) - r - np.log(scipy.special.factorial(k))
    return k * np.log(r) - r - scipy.special.gammaln(k+1)


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
    _tricky_alleles: np.ndarray = None

    def set_tricky_alleles(self, tricky_alleles):
        self._tricky_alleles = tricky_alleles

    def as_sparse(self):
        return SparseLimitedFrequencySamplingComboModel.from_non_sparse(self)

    def limit_to_n_individuals(self, n):
        for i, count in enumerate(self.diplotype_counts):
            self.diplotype_counts[i] = count[:, 0:n].copy()

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
        ret = cls([np.zeros((n_variants, max_count), dtype=np.uint16) for i in range(3)])
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
        #counts = counts.astype(np.float16)
        sums = np.sum(counts, axis=-1)[:, None]
        frequencies = np.log(counts / sums)
        poisson_lambda = (np.arange(counts.shape[1])[None,:] + error_rate) * base_lambda
        poisson_lambda = poisson_lambda.astype(np.float16)
        prob = fast_poisson_logpmf(observed_counts[:,None].astype(np.float16), poisson_lambda)
        prob = logsumexp(frequencies + prob, axis=-1)
        return prob

    @staticmethod
    def _logpmf2(observed_counts, counts, base_lambda, error_rate):
        # Identical to _logpmf2, some lower memory usage by combining stuff
        poisson_lambda = (np.arange(counts.shape[1])[None, :] + error_rate) * base_lambda
        poisson_lambda = poisson_lambda.astype(np.float16)

        prob = logsumexp(np.log(counts / np.sum(counts, axis=-1)[:, None]) +
                         fast_poisson_logpmf(observed_counts[:, None].astype(np.float16), poisson_lambda)
                         , axis=-1)
        # prob where there are no counts should be 0 (-inf)
        prob[np.sum(counts, axis=-1) == 0] = -np.inf
        return prob

    @staticmethod
    def _gpu_logpmf(observed_counts, counts, base_lambda, error_rate):
        try:
            import custats
            import cupy as cp
        except ImportError:
            logging.error("Cucounter and cupy must be installed to run GPU genotyping")
            raise

        logging.info("Putting in cuda memory")

        t0 = time.perf_counter()
        observed_counts = observed_counts.astype(np.int32)
        logging.info("As type took %.3f sec" % (time.perf_counter()-t0))
        t0 = time.perf_counter()
        observed_counts = cp.asanyarray(observed_counts)
        logging.info("Moving observed counts to cuda memory took %.3f sec" % (time.perf_counter()-t0))

        t0 = time.perf_counter()
        counts = counts.astype(np.float32)
        logging.info("changing counts dtype took %.3f sec" % (time.perf_counter()-t0))

        t0 = time.perf_counter()
        counts = cp.asanyarray(counts)
        logging.info("SHAPE/DTYPE counts: %s/%s" % (counts.shape, counts.dtype))
        logging.info("Moving counts to cuda memory took %.3f sec" % (time.perf_counter()-t0))

        res = custats.functions.experimental_logpmf(observed_counts, counts, base_lambda, error_rate)
        return cp.asnumpy(res)

    def logpmf(self, observed_counts, d, base_lambda=1.0, error_rate=0.01, gpu=False, n_threads=16):
        logging.info("Will use %d threads" % n_threads)
        logging.debug("base lambda in LimitedFreq model is %.3f" % base_lambda)
        logging.debug("Error rate is %.3f" % error_rate)
        logging.info("Will use GPU? %s" % gpu)
        t0 = time.perf_counter()
        counts = self.diplotype_counts[d]  # [:, 0:5]
        counts = counts.astype(np.float16)
        if gpu:
            logging.info("USING GPU")
            prob = LimitedFrequencySamplingComboModel._gpu_logpmf(observed_counts, counts, base_lambda, error_rate)
        else:
            prob = run_numpy_based_function_in_parallel(
                LimitedFrequencySamplingComboModel._logpmf2, n_threads, (observed_counts, counts, base_lambda, error_rate)
            )
        logging.info("Logpmf took %.4f sec" % (time.perf_counter()-t0))

        if self._tricky_alleles is not None:
            # if there are tricky alleles, we set the prob to 1/3
            prob[self._tricky_alleles] = np.log(1/3)

        return prob

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
        m0, m1, m2 = self.diplotype_counts

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

        if prior > 0:
            assert all([np.all(np.sum(c, axis=-1) > 0) for c in self.diplotype_counts])
        logging.debug("Filling empty data took %.2f sec " % (time.perf_counter()-t))

    def fill_empty_data2(self, prior=1.0):
        """
        Fills by interpolating counts.
        Assumes there's never more than one genotype that has missing data.
        if c0, c1, c2 are counts for having 0, 1 or 2 copies of the allele then
        c2 = c1 + (c1 - c0)
        c1 = c0 + (c2 - c0) / 2
        c0 = c1 - (c2 - c1)
        """
        # find expected counts for all
        m0, m1, m2 = self.diplotype_counts
        missing0 = np.all(m0 == 0, axis=1)
        missing1 = np.all(m1 == 0, axis=1)
        missing2 = np.all(m2 == 0, axis=1)

        expected_counts = []

        for diplotype in [0, 1, 2]:
            logging.info("Computing expected counts for genotype %d" % diplotype)
            m = self.diplotype_counts[diplotype]
            not_missing = np.where(np.sum(m, axis=-1) > 0)[0]
            expected = np.zeros(m.shape[0], dtype=float)
            expected[not_missing] = np.sum(np.arange(m.shape[1]) * m[not_missing], axis=-1) / np.sum(m[not_missing],
                                                                                                     axis=-1)
            expected_counts.append(expected)

        e0, e1, e2 = expected_counts
        expected_on_missing2 = e1[missing2] + (e1[missing2] - e0[missing2])
        expected_on_missing1 = e0[missing1] + (e2[missing1] - e0[missing1]) / 2
        expected_on_missing0 = e1[missing0] - (e2[missing0] - e1[missing0])

        max_count = m0.shape[1]-1
        m0[missing0, np.minimum(np.round(expected_on_missing0).astype(int), max_count)] = 1
        m1[missing1, np.minimum(np.round(expected_on_missing1).astype(int), max_count)] = 1
        m2[missing2, np.minimum(np.round(expected_on_missing2).astype(int), max_count)] = 1

    def has_no_data(self, idx, threshold=2):
        missing = [np.sum(c[idx]) == 0 for c in self.diplotype_counts]
        # if 2 out of 3 is missing data, return True
        return sum(missing) >= threshold

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


# logsumexp for RaggedArray
def ragged_array_logsumexp(array, axis=-1):
    assert axis == -1
    max_elem = array.max(axis=axis, keepdims=True)
    return max_elem.ravel() + np.log(np.sum(np.exp(array-max_elem), axis=axis))


@dataclass
class SparseObservedCounts:
    indexes: Tuple[np.ndarray]  # position of nonzero elements
    frequencies: np.ndarray  # the values in the matrix at indexes
    #row_lens: np.ndarray  # how many nonzero elements on each row
    max_counts: int

    def describe_node(self, node):
        return ", ".join("")

    @classmethod
    def from_nonsparse(cls, model_counts: np.ndarray):
        logging.info("Making sparse from nonsparse")
        indexes = np.nonzero(model_counts)
        frequencies = np.log(model_counts / model_counts.sum(axis=-1, keepdims=True))
        log_memory_usage_now("Got frequencies")
        frequencies = frequencies[indexes]
        indexes = (indexes[0].astype(np.uint32), indexes[1].astype(np.uint16))
        max_counts = model_counts.shape[1]
        #row_lens = np.bincount(indexes[0], minlength=model_counts.shape[0])
        return cls(indexes, frequencies, max_counts)

    @staticmethod
    def _logpmf(observed_counts, indexes_0, indexes_1, base_lambda, error_rate, max_counts, probs):
        if isinstance(observed_counts, str):
            observed_counts = object_from_shared_memory(observed_counts)

        assert len(indexes_0) > 0
        row_lens = np.bincount(indexes_0 - indexes_0[0], minlength=indexes_0[-1]-indexes_0[0]+1)
        assert len(row_lens) == indexes_0[-1]-indexes_0[0]+1
        assert np.all(row_lens > 0)
        p_lambda = (np.arange(max_counts) + error_rate) * base_lambda
        probs += fast_poisson_logpmf(observed_counts[indexes_0], p_lambda[indexes_1])
        ra = RaggedArray(probs, row_lens)
        return ragged_array_logsumexp(ra, axis=-1)

    def logpmf(self, observed_counts, base_lambda, error_rate, n_threads=1):
        if n_threads > 1:
            return self.parallel_logpmf(observed_counts, base_lambda, error_rate, n_threads)

        probs = self.frequencies
        indexes_0 = self.indexes[0]
        indexes_1 = self.indexes[1]
        assert indexes_0[0] == 0
        assert indexes_0[-1] == len(observed_counts)-1, indexes_0[-1]
        return SparseObservedCounts._logpmf(observed_counts, indexes_0, indexes_1, base_lambda, error_rate, self.max_counts, probs)

    def parallel_logpmf(self, observed_counts, base_lambda, error_rate, n_threads=4):
        observed_counts_name = object_to_shared_memory(observed_counts)
        # not true if not all rows have values
        assert len(observed_counts) == self.indexes[0][-1]+1  # last index should be last row

        splits = [int(i) for i in np.linspace(0, self.indexes[0][-1], min(n_threads+1, self.indexes[0][-1]+1))]
        splits[-1] += 1  # last index must be at end
        splits = np.searchsorted(self.indexes[0], splits, side='left')
        chunks = [(a, b) for a, b in zip(splits[0:-1], splits[1:])]

        results = run_numpy_based_function_in_parallel(SparseObservedCounts._logpmf, n_threads,
                                                       [observed_counts_name, self.indexes[0], self.indexes[1], base_lambda, error_rate, self.max_counts, self.frequencies],
                                                       chunks=chunks)
        remove_shared_memory(observed_counts_name, limit_to_session=True)
        assert len(results) == len(observed_counts), len(results)
        return results

        #chunks = shared_memory_wrapper.util.interval_chunks(0, len(self.frequencies), 16)
        # Make chunks that are splitted where new unique indexes start

        results = []
        for start, end in chunks:
            print(start, end)
            indexes_0 = self.indexes[0][start:end]
            indexes_1 = self.indexes[1][start:end]
            probs = self.frequencies[start:end]
            res = SparseObservedCounts._logpmf(observed_counts_name, indexes_0, indexes_1, base_lambda, error_rate, self.max_counts, probs)
            assert len(res) == end-start, "%d != %d" % (len(res), end-start)
            results.append(res)

        return np.concatenate(results)


class SparseLimitedFrequencySamplingComboModel(Model):
    def __init__(self, counts: List[SparseObservedCounts], tricky_alleles: np.ndarray = None):
        self._counts = counts
        self._tricky_alleles = tricky_alleles

    def set_tricky_alleles(self, tricky_alleles):
        self._tricky_alleles = tricky_alleles

    def logpmf(self, observed_counts, genotype, base_lambda=1.0, error_rate=0.01, gpu=False, n_threads=1):
        # todo:
        # if there are less than one individual for all genotypes at a variant, we are not able to predict
        # then set the prob to 1/3
        res = self._counts[genotype].logpmf(observed_counts, base_lambda, error_rate, n_threads)
        if self._tricky_alleles is not None:
            # if there are tricky alleles, we set the prob to 1/3
            res[self._tricky_alleles] = np.log(1/3)

        assert len(res) == len(observed_counts), len(res)
        return res

    @classmethod
    def from_non_sparse(cls, model):
        return cls([
            SparseObservedCounts.from_nonsparse(c) for c in model.diplotype_counts
        ])

    def describe_node(self, variant_id):
        description = ""
        #return description
        for count in range(3):
            description += "Having %d copies: " % count
            variant_indexes = np.where(self._counts[count].indexes[0] == variant_id)[0]
            counts = self._counts[count].indexes[1][variant_indexes]
            if len(variant_indexes) == 0:
                description += "No individuals"
            else:
                description += " ".join(
                    "%d: %.3f" % (c, np.exp(self._counts[count].frequencies[i])) for i, c in zip(variant_indexes, counts)
                )

            #description += ', '.join("%d: %.3f" % (i, self._counts[count].frequencies[variant_id, i]) for i in np.nonzero(self._counts[count].frequencies[variant_id])[0])
            description += "\n"
        return description

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

    def fill_empty_data(self, prior=None):
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

        """
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
        """
        # only correct cases where one is missin
        missing0 = missing0 & ~missing1 & ~missing2
        missing1 = missing1 & ~missing0 & ~missing2
        missing2 = missing2 & ~missing0 & ~missing1

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
