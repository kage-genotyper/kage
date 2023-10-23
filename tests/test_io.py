import numpy as np
import pytest
# from kage.io import VcfWriter
from kage.preprocessing.variants import Variants
from kage.io import write_vcf, VcfEntryWithSingleIndividualGenotypes, \
    convert_biallelic_numeric_genotypes_to_multialellic_string_genotypes, \
    write_multiallelic_vcf_with_biallelic_numeric_genotypes, SimpleVcfEntry
import bionumpy as bnp


@pytest.fixture
def variants():
    variants = Variants.from_entry_tuples(
        [
            ("chr1", "")
        ]
    )


def test_vcf_writer():
    variants = SimpleVcfEntry.from_entry_tuples([
        ("chr1", 4, "A", "T"),
        ("chr2", 10, "AAA", "A")
    ])

    genotypes = bnp.as_encoded_array(["0/1", "1/1"])
    write_vcf(variants, genotypes, "test.vcf", header="#someheader\n")


def test_vcf_writer2():
    variants = SimpleVcfEntry.from_entry_tuples([
        ("chr1", 4, "A", "T"),
        ("chr1", 10, "G", "T"),
        ("chr2", 10, "AAA", "A")
    ])

    genotypes = np.array([1, 2, 1, 1])
    n_alleles_per_variant = np.array([2, 3, 2])
    write_multiallelic_vcf_with_biallelic_numeric_genotypes(variants, genotypes,
                                                            "test.vcf",
                                                            n_alleles_per_variant,
                                                            header="#someheader\n")


def test_convert_biallelic_numeric_genotypes_to_multialellic_string_genotypes():

    numeric = np.array([1, 2, 1, 3, 1, 3, 1, 2])
    n_alleles_per_variant = np.array([2, 3, 2, 3, 3])
    genotypes = convert_biallelic_numeric_genotypes_to_multialellic_string_genotypes(n_alleles_per_variant, numeric)
    correct = ["0/0", "1/1", "0/1", "0/2", "2/2"]
    assert genotypes.tolist() == correct
