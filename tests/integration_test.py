from kage.analysis.genotype_accuracy import IndexedGenotypes2, GenotypeAccuracy
import numpy as np
np.seterr(all="ignore")
from kage.indexing.main import make_index
from kage.command_line_interface import genotype
from kage.util import make_args_for_genotype_command


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

    args = make_args_for_genotype_command("test_index.npz", reads)
    genotype(args)

    truth = IndexedGenotypes2.from_biallelic_vcf(truth)
    sample = IndexedGenotypes2.from_biallelic_vcf("test_genotypes.vcf")
    accuracy = GenotypeAccuracy(truth, sample, limit_to="all")

    recall = accuracy.recall()
    precision = accuracy.precision()
    f1_score = accuracy.f1()

    assert f1_score >= 0.959


if __name__ == "__main__":
    test_saccer3()
