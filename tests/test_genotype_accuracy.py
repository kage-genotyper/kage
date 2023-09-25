from kage.analysis.genotype_accuracy import GenotypeAccuracy, read_vcf_with_genotypes, normalize_genotype, \
    IndexedGenotypes, normalized_genotypes_to_haplotype_matrix
import bionumpy as bnp
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.preprocessing.variants import SimpleVcfEntry
import numpy as np


def test_read_vcf():
    data = read_vcf_with_genotypes("../example_data/vcf_with_genotypes.vcf")
    print(data)
    print(data.genotype)


def test_normalize_genotype():
    assert normalize_genotype("0/0") == "0/0"
    assert normalize_genotype("5|1") == "1/5"
    assert normalize_genotype(".|1") == "./1"
    assert normalize_genotype(".") == "./."
    assert normalize_genotype("0/1:0.5") == "0/1"


def test_genotype_accuracy():
    truth = IndexedGenotypes({
        "1": "0/0",
        "2": "0/1",
        "10": "1|0",
    })

    sample = IndexedGenotypes({
        "1": "1/0",
        "2": "0/1",
        "10": "0/0"
    })

    accuracy = GenotypeAccuracy(truth, sample)
    assert accuracy.false_negative == 1
    assert accuracy.true_positive == 1
    assert accuracy.true_negative == 0
    assert accuracy.false_negative == 1

    assert accuracy.recall() == 0.5
    assert accuracy.precision() == 0.5


def test_indexed_genotypes_from_multiallelic():
    variants = SimpleVcfEntry.from_entry_tuples([
        ("chr1", 6, "A", "T"),
        ("chr1", 10, "A", "T,C,G"),
        ("chr1", 15, "A", "ATT,ATTTT"),
        ("chr1", 20, "C", "CTA"),
        ("chr1", 30, "A", "T")
    ])

    haplotype_matrix = SparseHaplotypeMatrix.from_nonsparse_matrix([
        [0, 1],
        [1, 3],
        [0, 2],
        [1, 1],
        [127, 1]
    ])

    indexed_genotypes = IndexedGenotypes.from_multiallelic_variants_and_haplotype_matrix(variants, haplotype_matrix)
    print(indexed_genotypes._index)
    assert indexed_genotypes["chr1-6-A-T"] == "0/1"
    assert indexed_genotypes["chr1-10-A-T"] == "0/1"
    assert indexed_genotypes["chr1-10-A-C"] == "0/0"
    assert indexed_genotypes["chr1-10-A-G"] == "0/1"
    assert indexed_genotypes["chr1-16--TT"] == "0/0"
    assert indexed_genotypes["chr1-16--TTTT"] == "0/1"
    assert indexed_genotypes["chr1-21--TA"] == "1/1"
    assert indexed_genotypes["chr1-30-A-T"] == "1/."  # 127 is encoding for missing
    print(indexed_genotypes)


def test_normalized_genotypes_to_haplotype_matrix():
    missing_genotype_encoding = 127

    genotypes = ["0/1", "./1", "0/0", "0/."]
    correct = [
        [0, 1],
        [missing_genotype_encoding, 1],
        [0, 0],
        [0, missing_genotype_encoding]
    ]

    matrix = normalized_genotypes_to_haplotype_matrix(genotypes, encode_missing_as=missing_genotype_encoding)
    assert np.all(matrix == correct)

