from kage.preprocessing.variants import VariantPadder, Variants
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



def test_variant_padder2(variants2, reference2):
    pass
