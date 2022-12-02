import pytest
from kage.models.helper_model import get_population_priors
import numpy as np

@pytest.fixture
def genotype_combo_matrix():
    return np.array([[[70, 20, 10],
                      [5, 5, 0],
                      [0, 0, 1]]])

@pytest.fixture
def helper_posterior():
    return np.array([[[0.5], [0.3], [0.2]]])


def _test_get_population_priors(genotype_combo_matrix):
    priors = get_population_priors(genotype_combo_matrix, weight=1, hyper_prior=0)
    true = np.array([[0.70, 0.20, 0.10],
                     [0.5, 0.5, 0],
                     [0, 0, 1]])
    assert np.all(priors==true)

def _test_get_population_priors_w_hyper_prior(genotype_combo_matrix):
    priors = get_population_priors(genotype_combo_matrix, weight=1, hyper_prior=0.1*np.eye(3))
    true = np.array([[70.1/100.1, 20/100.1, 10/100.1],
                     [5/10.1, 5.1/10.1, 0],
                     [0, 0, 1]])
    assert np.all(priors==true)



def _test_apply_priors(genotype_combo_matrix, helper_posterior):
    p = apply_priors(genotype_combo_matrix, 0, helper_posterior)
    true = np.array([[[.35, .10, .05],
                      [0.15, 0.15, 0],
                      [0, 0, 0.2]]])
    print(p)
    print(true)
    assert np.all(p==true)
