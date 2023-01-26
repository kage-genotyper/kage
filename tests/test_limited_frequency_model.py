import npstructures
from numpy.testing import assert_equal
import logging
logging.basicConfig(level=logging.DEBUG)
import time
from kage.models.sampling_combo_model import LimitedFrequencySamplingComboModel, SparseObservedCounts
import numpy as np
#import cupy as np
import scipy
from scipy.special import logsumexp
from shared_memory_wrapper.shared_memory import run_numpy_based_function_in_parallel
from kage.util import log_memory_usage_now
from npstructures import RaggedArray
#npstructures.set_backend(np)
import warnings
warnings.filterwarnings('ignore')


def logsumexp2(array, axis=-1):
    assert axis == -1
    max_elem = array.max(axis=axis, keepdims=True)
    return max_elem.ravel() + np.log(np.sum(np.exp(array-max_elem), axis=axis))


def test_logsumexp2():
    array = np.log(np.arange(12)).reshape(3, 4)
    assert_equal(logsumexp(array, axis=-1), logsumexp2(array, axis=-1))

def fast_poisson_logpmf(k, r):
    # should return the same as scipy.stats.logpmf(k, r), but is  ~4x faster
    #return k  * np.log(r) - r - np.log(scipy.special.factorial(k))
    return k * np.log(r) - r - scipy.special.gammaln(k+1)

def func(observed_counts, counts, base_lambda, error_rate):
    sums = np.sum(counts, axis=-1)[:, None]
    frequencies = np.log(counts / sums)
    poisson_lambda = (np.arange(counts.shape[1])[None, :] + error_rate) * base_lambda
    prob = fast_poisson_logpmf(observed_counts[:, None], poisson_lambda)
    prob = logsumexp(frequencies + prob, axis=-1)
    return prob


def func_2(observed_counts, frequencies_indexes, probs, base_lambda, error_rate, row_len, table):
    row_lens = np.bincount(frequencies_indexes[0], minlength=len(observed_counts))
    p_lambda = (np.arange(row_len) + error_rate) * base_lambda
    probs += fast_poisson_logpmf(observed_counts[frequencies_indexes[0]], p_lambda[frequencies_indexes[1]])
    ra = RaggedArray(probs, row_lens)
    return logsumexp2(ra, axis=-1)


def func_3(observed_counts, model: SparseObservedCounts, base_lambda, error_rate):
    row_lens = np.bincount(model.indexes[0], minlength=len(observed_counts))
    p_lambda = (np.arange(model.max_counts) + error_rate) * base_lambda
    probs = model.frequencies
    probs += fast_poisson_logpmf(observed_counts[model.indexes[0]], p_lambda[model.indexes[1]])
    ra = RaggedArray(probs, row_lens)
    return logsumexp2(ra, axis=-1)


def test_and_profile_logpmf(n_counts=50000, row_len=15):
    # observed_counts are counts on a node, typically 0-20
    # these we have in GPU-memory from counting
    observed_counts = np.random.randint(0, 20, n_counts, dtype=np.uint32)

    # expected counts in our model
    # is a matrix with n_counts rows and 5 columns
    # each column represents the average number of individuals
    # with that amount of counts in our model
    # can be float16 or float32, rows should not sum to 0
    model_counts = np.zeros((n_counts, row_len), dtype=np.float16)
    model_counts[:, 2] = np.random.randint(1, 100, n_counts) / 10
    # model_counts = (np.random.randint(0, 100, (n_counts, 5)) / 10).astype(np.float16) + 0.1
    
    base_lambda = 7.5  # float
    error_rate = 0.01  # float


    indexes = np.nonzero(model_counts)
    table = fast_poisson_logpmf(np.arange(2500)[:, np.newaxis], (np.arange(row_len)[None, :] + error_rate) * base_lambda)
    frequencies = np.log(model_counts/model_counts.sum(axis=-1, keepdims=True))
    frequencies = frequencies[indexes]
    t = time.perf_counter()
    result = func_2(observed_counts, indexes, frequencies, base_lambda, error_rate, row_len, table)
    print('t1', time.perf_counter()-t)

    model = SparseObservedCounts.from_nonsparse(model_counts)
    model_copy = SparseObservedCounts.from_nonsparse(model_counts)
    # paralell
    t = time.perf_counter()
    result_parallel = model.parallel_logpmf(observed_counts, base_lambda, error_rate, n_threads=16)
    print("Time parallel:", time.perf_counter()-t)
    print(result_parallel, result)
    print(len(result_parallel), len(result))
    #assert False
    assert_equal(result_parallel, result)

    t = time.perf_counter()
    #result2 = func_3(observed_counts, model, base_lambda, error_rate)
    result2 = model_copy.logpmf(observed_counts, base_lambda, error_rate)
    print(result, result2)
    print('t3', time.perf_counter()-t)
    assert_equal(result, result2)


    t = time.perf_counter()    
    #r2 = func(observed_counts, model_counts, base_lambda, error_rate)
    r2 = run_numpy_based_function_in_parallel(func, 16, [observed_counts, model_counts, base_lambda, error_rate])
    print('t2', time.perf_counter()-t)
    #assert_equal(result[0:100], r2[0:100])



if __name__ == "__main__":
    #test_logsumexp2()
    test_and_profile_logpmf(2800000)
