import logging
import numpy as np
from scipy.stats import nbinom, poisson
from scipy.special import gamma, factorial, gammaln, logsumexp, hyp2f1, hyp1f1, hyperu
import matplotlib.pyplot as plt
COVERAGE=15

def special_sum(k, r, p):
    U = tricomis_function(-k, -k-r+1, 1/p)
    return p**k*gamma(r)*U/gamma(k+1)

def log_special_sum(k, r, p):
    logU = np.log(hyperu(-k, -k-r+1, 1/p))
    return k*np.log(p)+gammaln(r)+logU-gammaln(k+1)

class CombinationModel:
    def __init__(self, base_lambda, r, p, certain_repeats=0, p_sum=0):
        self._base_lambda = base_lambda
        self._r = r
        self._p = p
        self._certain_repeats = certain_repeats
        self._p_sum = p_sum

    @classmethod
    def from_p_sums(cls, base_lambda, p_sum, p_sq_sum):
        sum_is_sq_sum = p_sum==p_sq_sum
        alpha = (p_sum)**2/(p_sum-p_sq_sum)
        beta = p_sum/(base_lambda*(p_sum-p_sq_sum))
        return cls(base_lambda, alpha, 1/(1+beta), sum_is_sq_sum, p_sum)
        
    def simple_pmf(self, k, n_copies = 1):
        i = np.arange(k+1)
        probs = poisson.pmf(k-i, n_copies*self._base_lambda)*nbinom.pmf(i, self._r, 1-self._p)
        return probs.sum()

    def logpmf(self, k, n_copies=1):
        dummy_count = 0.03 * self._base_lambda
        mu, r, p = (n_copies * self._base_lambda + dummy_count, self._r, self._p)
        result = -r * np.log(p / (1 - p)) - mu + (r + k) * np.log(mu) - gammaln(k + 1) + np.log(hyperu(r, r + k + 1, mu / p))
        return np.where(self._certain_repeats, poisson.logpmf(k, (n_copies+self._p_sum)*self._base_lambda+dummy_count), result)

    """
    def logpmf(self, k, n_copies = 1):
        mu = n_copies*self._base_lambda
        outside_log_sum = self._r*np.log(1-self._p)-mu-gammaln(self._r)+k*np.log(mu)
        s_log_sum = log_special_sum(k, self._r, self._p/mu)
        r = outside_log_sum+s_log_sum
        logging.info(":....................")
        logging.info(r.shape)
        logging.info(self._r.shape)
        logging.info(k.shape)
        return np.where(np.isnan(self._r), poisson.logpmf(k, n_copies*self._base_lambda), r)
    """

    def pmf(self, k, n_copies = 1):
        return np.exp(self.logpmf(k, n_copies))



class CombinationModelBothAlleles:
    def __init__(self, model_ref, model_alt):
        self._model_ref = model_ref
        self._model_alt = model_alt

    def pmf(self, k1, k2, genotype):
        ref_probs = self._model_ref.logpmf(k1, genotype)
        alt_probs = self._model_alt.logpmf(k2, 2-genotype)
        #result = np.exp(ref_probs+alt_probs)
        result = ref_probs + alt_probs
        return result

    def logpmf(self, k1, k2, genotype):
        return self.pmf(k1, k2, genotype)


def simulate(alpha, beta, base_lambda, n=1000000):
    mu = np.random.gamma(alpha, 1/beta, n)
    of = np.random.poisson(mu)
    ot = np.random.poisson(base_lambda, n)
    o = of+ot
    x = np.arange(10)
    observed = [np.sum(o==i)/n for i in x]
    model = CombinationModel(base_lambda, alpha, 1/(1+beta))
    predicted = [model.logpmf(i) for i in x]
    print(predicted)
    plt.bar(np.arange(10), observed, color=(0, 1, 0))
    plt.bar(np.arange(10), predicted, color=(0, 0, 1, 0.5));plt.show()
    return o

def simulate_from_ps(ps, base_lambda, n=100000, genotype=1):
    ps = np.asanyarray(ps)
    n_copies = np.array([np.sum(np.random.rand(ps.size)<ps) for _ in range(n)])
    of = np.random.poisson(n_copies*base_lambda)
    ot = np.random.poisson(base_lambda*genotype, n)
    o = of+ot
    x = np.arange(10)
    observed = [np.sum(o==i)/n for i in x]
    model = CombinationModel.from_p_sums(base_lambda, ps.sum(), np.sum(ps**2))
    predicted = [model.logpmf(i, genotype) for i in x]
    plt.bar(np.arange(10), observed, color=(0, 1, 0))
    plt.bar(np.arange(10), predicted, color=(0, 0, 1, 0.5));plt.show()
    return o


"""
sim_ps = np.random.rand(3)
o = simulate_from_ps(sim_ps, 1.2);

ps = np.array([0.1, 0.5, 0.7])
model = CombinationModel.from_p_sums(COVERAGE, ps.sum(), np.sum(ps**2))
"""
if False: 
    print(model.simple_pmf(10))
    print(model.pmf(10))
    
    print(model.log_pmf(10))
    print(model.first_intermediate_pmf(10))
    
    print(model.intermediate_pmf(10))
    print(model.intermediate2_pmf(10))
    print(model.intermediate2_logpmf(10))
                         
