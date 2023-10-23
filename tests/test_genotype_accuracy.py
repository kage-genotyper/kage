from kage.analysis.genotype_accuracy import GenotypeAccuracy, read_vcf_with_genotypes, normalize_genotype, \
    IndexedGenotypes, normalized_genotypes_to_haplotype_matrix, MultiAllelicVariant, IndexedGenotypes2, \
    IndexedGenotypes3
import bionumpy as bnp
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.io import SimpleVcfEntry
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
        "10": "0/1",
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

    assert accuracy.confusion_matrix["hetero"] == 2
    assert accuracy.concordance_hetero == 0.5
    assert accuracy.concordance_homo_alt == 0.0
    assert accuracy.concordance_homo_ref == 0.0

    assert accuracy.confusion_matrix["homo_ref"] == 1
    assert accuracy.confusion_matrix["homo_alt"] == 0

    assert accuracy.weighted_concordance == 0.5 / 3


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

    indexed_genotypes_multiallelic = IndexedGenotypes.from_multiallelic_variants_and_haplotype_matrix_without_biallelic_converion(variants, haplotype_matrix)
    assert indexed_genotypes_multiallelic["chr1-6-A-T"] == "0/1"
    assert indexed_genotypes_multiallelic["chr1-10-A-C-G-T"] == "2/3"
    assert indexed_genotypes_multiallelic["chr1-30-A-T"] == "1/."  # 127 is encoding for missing

    # test v2
    indexed_genotypes2 = IndexedGenotypes2.from_multiallelic_variants_and_haplotype_matrix_without_biallelic_converion(variants, haplotype_matrix)
    print(indexed_genotypes2._index)
    assert indexed_genotypes2["chr1-10-A"] == MultiAllelicVariant("chr1", 10, "A", ["T", "C", "G"], [1, 3])


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



def test_multiallelic_variant():
    ref = MultiAllelicVariant("chr1", 10, "A", ["T", "C", "G"], (0, 1))
    other = MultiAllelicVariant("chr1", 10, "A", ["C", "G"], (0, 1))

    other.normalize_using_reference_variant(ref)
    assert other.genotype == [0, 2]

    other = MultiAllelicVariant("chr1", 10, "A", ["C", "G"], (127, 1))
    other.normalize_using_reference_variant(ref)
    assert other.genotype == [127, 2]  # missing is first when string sorting

    other = MultiAllelicVariant("chr1", 10, "A", ["C", "G"], (127, 2))
    other.normalize_using_reference_variant(ref)
    assert other.genotype == [127, 3]

    ref = MultiAllelicVariant("chr2", 4, "A", ["ACTACTACACT", "AAAA", "T"], (0, 0))
    other = MultiAllelicVariant("chr2", 4, "A", ["T"], (1, 0))
    other.normalize_using_reference_variant(ref)
    assert other.alt_sequences == ref.alt_sequences
    assert other.genotype == [0, 3]


def test_variant_type():

    variant = MultiAllelicVariant("chr1", "10", "A", "T", (0, 0))
    assert variant.type() == "snp"

    variant = MultiAllelicVariant("chr1", "10", "AT", "A", (0, 0))
    assert variant.type() == "indel"

    variant = MultiAllelicVariant("chr1", "10", "A", ["A", "TT"] , (0, 0))
    assert variant.type() == "indel"

    variant = MultiAllelicVariant("chr1", "10", "A", ["A"*50, "TT"] , (0, 0))
    assert variant.type() == "sv"


def test_read_vcf_with_missing_genotypes():
    truth = IndexedGenotypes2.from_multiallelic_vcf("vcf_with_missing.vcf", convert_to_biallelic=False)

    i = 0
    for id, variant in truth.items():
        print(id, variant.genotype_string())

        if i > 10:
            break
        i += 1

    truth2 = IndexedGenotypes3.from_multiallelic_vcf("vcf_with_missing.vcf", convert_to_biallelic=False)

    i = 0
    for id, variant in truth.items():
        print(id, variant.genotype_string())

        if i > 10:
            break
        i += 1

    comparison = GenotypeAccuracy(truth, truth)

    print(comparison.confusion_matrix)
