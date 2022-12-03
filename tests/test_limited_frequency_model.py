import time
from kage.models.sampling_combo_model import LimitedFrequencySamplingComboModel
import numpy as np
import scipy
from scipy.special import logsumexp


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


def profile_logpmf(n_counts=50000):
    # observed_counts are counts on a node, typically 0-20
    # these we have in GPU-memory from counting
    observed_counts = np.random.randint(0, 20, n_counts, dtype=np.uint32)

    # expected counts in our model
    # is a matrix with n_counts rows and 5 columns
    # each column represents the average number of individuals
    # with that amount of counts in our model
    # can be float16 or float32, rows should not sum to 0
    model_counts = (np.random.randint(0, 100, (n_counts, 5)) / 10).astype(np.float16) + 0.1

    base_lambda = 7.5  # float
    error_rate = 0.01  # float

    t = time.perf_counter()
    result = func(observed_counts, model_counts, base_lambda, error_rate)
    print(time.perf_counter()-t)


if __name__ == "__main__":
    profile_logpmf(28000000)
