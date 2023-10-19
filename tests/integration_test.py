from collections import namedtuple
import pytest
from kage.analysis.genotype_accuracy import IndexedGenotypes2, GenotypeAccuracy
from kage.simulation.simulation import run_genotyper_on_simualated_data
from kage.genotyping.combination_model_genotyper import CombinationModelGenotyper
import numpy as np
np.seterr(all="ignore")
import random
from kage.indexing.main import make_index
from kage.command_line_interface import genotype


@pytest.mark.skip(reason="outdated")
def test_simple():
    np.random.seed(1)
    random.seed(1)

    correct_rate = run_genotyper_on_simualated_data(CombinationModelGenotyper, n_variants=1000, n_individuals=2000,
                                                    duplication_rate=0.03, average_coverage=15, coverage_std=0.5)

    assert correct_rate >= 0.980, "Correct rate on trivial genotyping dropped" \
                                  " below 0.980 and is %.5f. Something may be wrong." % correct_rate


def test_saccer3():
    """
    Tests full indexing and genotyping using CLI on a small sample
    """

    ref = "test_data_sacCer3/reference.fa"
    vcf = "test_data_sacCer3/filtered_population.vcf.gz"
    reads = "test_data_sacCer3/reads.fq.gz"
    truth = "test_data_sacCer3/truth.vcf"

    make_index(ref, vcf, "test_index.npz",
               k=31, modulo=200000033,
               variant_window=5,
               n_threads=4)

    Args = namedtuple("args", ["index_bundle",
                               "reads",
                               "out_file_name",
                               "kmer_size",
                               "average_coverage",
                               "debug",
                               "n_threads",
                               "use_naive_priors",
                               "ignore_helper_model",
                               "ignore_helper_variants",
                               "min_genotype_quality",
                               "sample_name_output",
                               "ignore_homo_ref",
                               "do_not_write_genotype_likelihoods",
                               "gpu",
                               "counts"])
    args = Args("test_index.npz", reads, "test_genotypes.vcf", 31, 15, False, 4,
                False, False, False, 0, "sample", False, True, False, None)
    genotype(args)

    truth = IndexedGenotypes2.from_biallelic_vcf(truth)
    sample = IndexedGenotypes2.from_biallelic_vcf("test_genotypes.vcf")
    accuracy = GenotypeAccuracy(truth, sample, limit_to="all")

    recall = accuracy.recall()
    precision = accuracy.precision()
    f1_score = accuracy.f1()

    assert f1_score >= 0.98


if __name__ == "__main__":
    test_saccer3()
