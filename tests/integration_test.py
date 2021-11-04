from kage.simulation import run_genotyper_on_simualated_data
from kage.combination_model_genotyper import CombinationModelGenotyper
import numpy as np
import random

def test_simple():
    np.random.seed(1)
    random.seed(1)

    correct_rate = run_genotyper_on_simualated_data(CombinationModelGenotyper, n_variants=1000, n_individuals=2000,
                                                    duplication_rate=0.1, average_coverage=15, coverage_std=1.0)


    assert correct_rate >= 0.999, "Correct rate on trivial genotyping dropped" \
                                  " below 0.999 and is %.5f. Something may be wrong." % correct_rate

test_simple()