import numpy as np
from scipy.stats import nbinom, poisson
from scipy.special import gamma, factorial, gammaln, logsumexp, hyp2f1, hyp1f1
COVERAGE=15

def special_sum(k, r, p):
    U = tricomis_function(-k, -k-r+1, 1/p)
    return p**k*gamma(r)*U/gamma(k+1)

def log_special_sum(k, r, p):
    logU = log_tricomis_function(-k, -k-r+1, 1/p)
    return k*np.log(p)+gammaln(r)+logU-gammaln(k+1)

def tricomis_function(a, b, z):
    M = hyp1f1
    return gamma(1-b)/gamma(a+1-b)*M(a,b,z)+gamma(b-1)/gamma(a)*z**(1-b)*M(a+1-b, 2-b, z)

def log_tricomis_function(a, b, z):
    M = hyp1f1
    return logsumexp([gammaln(1-b)-gammaln(a+1-b)+np.log(M(a,b,z)), gammaln(b-1)-gammaln(a)+(1-b)*np.log(z)+np.log(M(a+1-b, 2-b, z))])



class CombinationModel:
    def __init__(self, coverage, p_sum, p_sq_sum):
        self._base_lambda = coverage
        mu = p_sum*coverage
        var = p_sq_sum
        self._alpha = (p_sum)**2/p_sq_sum
        self._beta = p_sum/p_sq_sum
        self._r = self._alpha
        self._p = 1/(1+self._beta)
        
    def pmf(self, k, n_copies = 1):
        i = np.arange(k+1)
        #probs = poisson.pmf(i, n_copies*self._base_lambda)*nbinom.pmf(k-i, self._r, self._p)
        probs = poisson.pmf(k-i, n_copies*self._base_lambda)*nbinom.pmf(i, self._r, 1-self._p)
        return probs.sum()

    def log_pmf(self, k, n_copies = 1):
        i = np.arange(k+1)
        log_probs = poisson.logpmf(i, n_copies*self._base_lambda)+nbinom.logpmf(k-i, self._r, self._p)
        return np.exp(logsumexp(log_probs))

    def first_intermediate_pmf(self, k, n_copies = 1):
        mu = n_copies*self._base_lambda
        i = np.arange(k+1)
        s = mu**(k-i)/factorial(k-i)*np.exp(-mu)*gamma(i+self._r)/gamma(self._r)/factorial(i)*self._p**self._r*(1-self._p)**i
        return s.sum()

    def second_intermediate_pmf(self, k, n_copies = 1):
        mu = n_copies*self._base_lambda
        i = np.arange(k+1)
        s = mu**(k-i)/factorial(k-i)*np.exp(-mu)*gamma(i+self._r)/gamma(self._r)/factorial(i)*self._p**self._r*(1-self._p)**i
        return s.sum()

    def intermediate_pmf(self, k, n_copies = 1):
        lambd = n_copies*self._base_lambda
        outside_sum = (1-self._p)**self._r*np.exp(-lambd)/gamma(self._r)*lambd**k
        i = np.arange(k+1)
        s = gamma(i+self._r)/factorial(i)/factorial(k-i)*(self._p/lambd)**i
        return outside_sum*s.sum()

    def intermediate2_pmf(self, k, n_copies = 1):
        mu = n_copies*self._base_lambda
        outside_sum = (1-self._p)**self._r*np.exp(-mu)/gamma(self._r)*mu**k
        i = np.arange(k+1)
        s = gamma(i+self._r)/factorial(i)/factorial(k-i)*(self._p/mu)**i
        s_sum = special_sum(k, self._r, self._p/mu)
        return outside_sum*s_sum

    def pmf(self, k, n_copies = 1):
        mu = n_copies*self._base_lambda
        outside_log_sum = self._r*np.log(1-self._p)-mu-gammaln(self._r)+k*np.log(mu)
        i = np.arange(k+1)
        s_log_sum = log_special_sum(k, self._r, self._p/mu)
        return np.exp(outside_log_sum+s_log_sum)

    def intermediate_log_pmf(self, k, n_copies = 1):
        lambd = n_copies*self._base_lambda
        outside_log_sum = self._r*np.log(1-self._p)-lambd-gammaln(self._r)+k*np.log(lambd)
        i = np.arange(k+1)
        log_s = gammaln(i+self._r)-gammaln(i+1)-gammaln(k-i+1)+i*np.log(self._p/lambd)
        return np.exp(outside_log_sum+logsumexp(log_s))


ps = np.array([0.1, 0.5, 0.7])
model = CombinationModel(COVERAGE, COVERAGE*ps.sum(), np.sum((COVERAGE*ps)**2))
print(model.pmf(10))
print(model.log_pmf(10))
print(model.first_intermediate_pmf(10))
print(model.intermediate_pmf(10))
print(model.intermediate2_pmf(10))
print(model.intermediate2_logpmf(10))
                         
