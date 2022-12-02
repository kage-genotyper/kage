from kage.simulation.simulation import run_genotyper_on_simualated_data
from kage.genotyping.combination_model_genotyper import CombinationModelGenotyper
import numpy as np
np.seterr(all="ignore")
import random

def test_simple():
    np.random.seed(1)
    random.seed(1)

    correct_rate = run_genotyper_on_simualated_data(CombinationModelGenotyper, n_variants=1000, n_individuals=2000,
                                                    duplication_rate=0.03, average_coverage=15, coverage_std=0.5)


    assert correct_rate >= 0.980, "Correct rate on trivial genotyping dropped" \
                                  " below 0.980 and is %.5f. Something may be wrong." % correct_rate

test_simple()