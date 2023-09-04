import logging
logging.basicConfig(level=logging.INFO)
import pytest
import numpy as np
import bionumpy as bnp
from kage.preprocessing.variants import Variants, SimpleVcfEntry


def test_variants_from_multialellic_vcf():
    vcf = bnp.open("tests/multiallelic.vcf").read()
    variants = Variants.from_multiallelic_vcf_entry(vcf)
    print(variants)

    correct = Variants.from_entry_tuples([
        ("chr1", 4, "A" , "T"),
        ("chr1", 4, "A", "G"),
        ("chr1", 7, "G", "T"),
        ("chr1", 10, "GG", ""),
        ("chr1", 10, "GG", "TG")
    ])

    assert np.all(variants == correct)


def test_variants_from_multiallelic_vcf_entry():
    entry = SimpleVcfEntry.from_entry_tuples(
        [
            ("chr1", 2, "ACTG", "A"),
            ("chr1", 10, "A", "AT,CACA,ACATA,ACA"),
            ("chr2", 5, "G", "T")
        ]
    )
    variants = Variants.from_multiallelic_vcf_entry(entry)
    correct = Variants.from_entry_tuples(
        [
            ("chr1", 3, "CTG", ""),
            ("chr1", 11, "", "T"),
            ("chr1", 11, "", "ACA"),
            ("chr1", 11, "", "CATA"),
            ("chr1", 11, "", "CA"),
            ("chr2", 5, "G", "T")
        ]
    )
    assert np.all(variants == correct)
