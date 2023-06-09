from kage.preprocessing.variants import VariantPadder, Variants, get_padded_variants_from_vcf
import pytest
from bionumpy.datatypes import VCFEntry
import bionumpy as bnp
import numpy as np


@pytest.fixture
def vcf_entry():
    return VCFEntry.from_entry_tuples(
        [
            ("1", 4, "ID", "A", "C", ".", "PASS", "."),
            ("1", 8, "ID", "AAAA", "A", ".", "PASS", "."),
            ("1", 9, "ID", "A", "G", ".", "PASS", "."),
            ("1", 12, "ID", "A", "ATTT", ".", "PASS", "."),
        ]
    )


@pytest.fixture
def variants(vcf_entry):
    return Variants.from_vcf_entry(vcf_entry)


def test_variant_from_vcf_entry(variants):
    assert variants.ref_seq.tolist() == ["A", "AAA", "A", ""]
    assert variants.alt_seq.tolist() == ["C", "", "G", "TTT"]
    assert np.all(variants.position == [4, 9, 9, 13])


@pytest.fixture
def reference():
    return bnp.as_encoded_array("A"*16)


def test_reference_mask(variants, reference):
    padder = VariantPadder(variants, reference)
    mask = padder.get_reference_mask()
    assert np.all(mask == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0])


def test_padding_distance(variants, reference):
    padder = VariantPadder(variants, reference)
    dist = padder.get_distance_to_ref_mask(dir="left")
    assert np.all(dist == [0,0,0,0,1,0,0,0,0,1,2,3,0,0])

    dist = padder.get_distance_to_ref_mask(dir="right")
    assert np.all(dist == [0,0,0,0,1,0,0,0,0,3,2,1,0,0])

    print(dist)


def test_variant_padder(variants, reference):
    padder = VariantPadder(variants, reference)
    padded = padder.run()

    correct = Variants.from_entry_tuples(
        [("1", 4, "A", "C"),
         ("1", 9, "AAA", ""),
         ("1", 9, "AAA", "GAA"),
         ("1", 13, "", "TTT")
         ]
    )
    print(padded)

    assert padded == correct


@pytest.fixture
def variants2():
    return Variants.from_entry_tuples(
        [
            ("1", 4, "TGCAT", ""),
            ("1", 6, "C", "G"),
            ("1", 7, "A", "T"),
            ("1", 8, "T", "C"),
        ]
    )

@pytest.fixture
def reference2():
    return bnp.as_encoded_array("TGCA"*4)


def test_variant_padder2(variants2, reference2):
    padder = VariantPadder(variants2, reference2)
    padded = padder.run()

    correct = Variants.from_entry_tuples(
        [
            ("1", 4, "TGCAT", ""),
            ("1", 4, "TGCAT", "TGGAT"),
            ("1", 4, "TGCAT", "TGCTT"),
            ("1", 4, "TGCAT", "TGCAC"),
        ]
    )
    print(padded)

    assert padded == correct



def test_get_padded_variants_from_vcf():
    variants = get_padded_variants_from_vcf("example_data/few_variants_two_chromosomes.vcf", "example_data/small_reference_two_chromosomes.fa")
    print(variants)