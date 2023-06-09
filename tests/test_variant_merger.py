from kage.preprocessing.variants import VariantMerger
import pytest
from bionumpy.datatypes import VCFEntry
import bionumpy as bnp
import numpy as np


@pytest.fixture
def variants():
    return VCFEntry.from_entry_tuples(
        [
            ("1", 4, "ID", "A", "C", ".", "PASS", "."),
            ("1", 8, "ID", "AAAA", "A", ".", "PASS", "."),
            ("1", 9, "ID", "A", "G", ".", "PASS", "."),
            ("1", 12, "ID", "A", "ATTT", ".", "PASS", "."),
        ]
    )

@pytest.fixture
def reference():
    return bnp.as_encoded_array("A"*16)


def test_reference_mask(variants, reference):
    merger = VariantMerger(variants, reference)
    mask = merger.get_reference_mask()
    #assert np.all(mask == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1])


def test_variant_merger(variants, reference):
    merger = VariantMerger(variants, reference)
    merged = merger.merge()
