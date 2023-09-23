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


@pytest.mark.skip("Outdated")
def test_reference_mask(variants, reference):
    padder = VariantPadder(variants, reference)
    mask = padder.get_reference_mask()
    assert np.all(mask == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0])


def test_padding_distance(variants, reference):
    padder = VariantPadder(variants, reference)
    dist = padder.get_distance_to_ref_mask(dir="left")
    assert np.all(dist == [0,0,0,0,0,0,0,0,0,0,1,2,0,0])

    dist = padder.get_distance_to_ref_mask(dir="right")
    assert np.all(dist == [0,0,0,0,0,0,0,0,0,2,1,0,0,0])

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


def test_variant_padder_with_insertion():
    """
    Variants(chromosome=encoded_array('chrI'), position=array(33673), ref_seq=encoded_array(''), alt_seq=encoded_array('CCGCTAAC'))
    Variants(chromosome=encoded_array('chrI'), position=array(33673), ref_seq=encoded_array('CT'), alt_seq=encoded_array('TT'))
    """
    variants = Variants.from_entry_tuples([
        ("chr1", 5, "", "ACGT"),
        ("chr1", 5, "AA", "TT")
    ])
    reference = bnp.as_encoded_array("A"*16)
    padder = VariantPadder(variants, reference)
    padded = padder.run()
    correct = Variants.from_entry_tuples([
        ("chr1", 5, "", "ACGT"),
        ("chr1", 5, "AA", "TT")
    ])
    assert correct == padded
    print(padded)


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


def test_variant_padder_with_deletion():
    ref = bnp.as_encoded_array("AAAAAGGGGGGAAAA")
    variants = Variants.from_entry_tuples([
        ("1", 5, "GGGGGG", ""),
        ("1", 7, "G", "T")
    ])
    padder = VariantPadder(variants, ref)
    padded = padder.run()

    correct = Variants.from_entry_tuples([
        ("1", 5, "GGGGGG", ""),
        ("1", 5, "GGGGGG", "GGTGGG")
    ])

    print(padded)
    assert padded == correct


def test_variant_padder_case():
    # checks previous bug with deletion that made hole in mask
    seq = "ATAAAGAACGGAAGGGGTTTAATAGTTGTATGC"
    ref = bnp.as_encoded_array("AAA" "A" + seq + "CCC")
    variants = Variants.from_entry_tuples([
        ("1", 4, seq, ""),
        ("1", 12, "C", "G"),
        ("1", 14, "", "CCGAGAT" )
    ])
    padder = VariantPadder(variants, ref)
    #mask = padder.get_mask_of_consecutive_ref_bases(dir="right")
    #print(padder.get_mask_of_consecutive_ref_bases(dir="left"))
    #print(padder.get_distance_to_ref_mask(dir="left"))
    correct = Variants.from_entry_tuples([
        ("1", 4, seq, ""),
        ("1", 4, seq, "ATAAAGAAGGGAAGGGGTTTAATAGTTGTATGC"),
        ("1", 4, seq, "ATAAAGAACG" "CCGAGAT" "GAAGGGGTTTAATAGTTGTATGC"),
    ])
    #print(padder.get_distance_to_ref_mask(dir="right"))
    padded = padder.run()

    #print(padded)
    print(padded.alt_seq[2].to_string())
    assert padded == correct


def test_pad_overlapping_variants():
    variants = Variants.from_entry_tuples([
        ("1", 4, "AAAAA", ""),
        ("1", 6, "AAAAA", ""),
        ("1", 7, "", "CCCCCCC"),
        ("1", 8, "AAAAA", ""),
        ("1", 14, "AAAAA", ""),
        ("1", 16, "AAAAA", ""),
        ("1", 18, "AAAAA", ""),
    ])
    ref = bnp.as_encoded_array("A"*50)
    padder = VariantPadder(variants, ref)
    padded = padder.run()
    correct = Variants.from_entry_tuples(
        [
            ("1", 4, "A" * 9, "AAAA"),
            ("1", 4, "A" * 9, "AAAA"),
            ("1", 4, "A" * 9, "AAACCCCCCCAAAAAA"),
            ("1", 4, "A" * 9, "AAAA"),
            ("1", 14, "A" * 9, "AAAA"),
            ("1", 14, "A" * 9, "AAAA"),
            ("1", 14, "A" * 9, "AAAA"),
        ]
    )
    print(padded)
    assert padded == correct


def test_pad_large_variants():
    variants = Variants.from_entry_tuples([
        ("chr1", 10, "A", "C"),
        ("chr1", 30, "", "T" * 10),
        ("chr1", 50, "G" * 10, ""),
        ("chr1", 70, "T", "C"),
        ("chr1", 100, "G" * 10, "")
    ])

    ref = bnp.as_encoded_array("A"*200000)
    padder = VariantPadder(variants, ref)
    padded = padder.run()
    print(padded)


def test_pad_neighbouring_snps():
    # neighbouring snps should not be joined
    variants = Variants.from_entry_tuples([
        ("chr1", 4, "A", "C"),
        ("chr1", 5, "A", "T")
    ])
    ref = bnp.as_encoded_array("A" * 10)
    padder = VariantPadder(variants, ref)
    padded = padder.run()

    # nothing should be padded here, previous error was that neighbouring variants god padded
    assert padded == variants
    print(padded)


def test_pad_small_del_with_snp_inside():
    variants = Variants.from_entry_tuples([
        ("chr1", 1, "AA", ""),
        ("chr1", 2, "A", "T")
    ])
    ref = bnp.as_encoded_array("AAAA")
    padder = VariantPadder(variants, ref)
    padded = padder.run()
    print(padded)

    correct = Variants.from_entry_tuples([
        ("chr1", 1, "AA", ""),
        ("chr1", 1, "AA", "AT")
    ])

    assert padded == correct


@pytest.fixture
def variants3():
    return Variants.from_entry_tuples([
        ("chr1", 1, "AA", ""),
        ("chr1", 1, "A", "T"),
        ("chr1", 3, "A", "G")
    ])


@pytest.fixture
def ref3():
    return bnp.as_encoded_array("AAAAA")


def test_pad_small_del_with_two_snps(variants3, ref3):
    # should not pad the second snp
    variants = variants3
    ref = ref3
    padder = VariantPadder(variants, ref)
    padded = padder.run()

    correct = Variants.from_entry_tuples([
        ("chr1", 1, "AA", ""),
        ("chr1", 1, "AA", "TA"),
        ("chr1", 3, "A", "G")
    ])
    print(padded)
    print(correct)
    assert padded == correct


def test_get_mask_of_consecutive_bases(variants3, ref3):
    padder = VariantPadder(variants3, ref3)
    mask = padder.get_mask_of_consecutive_ref_bases(dir='right')
    correct = [False, True, False, False, False]
    assert np.all(mask == correct)

    # left
    mask = padder.get_mask_of_consecutive_ref_bases(dir='left')
    correct = [False, False, True, False, False]
    assert np.all(mask == correct)


def test_get_dist_to_ref_mask_new(variants3, ref3):
    padder = VariantPadder(variants3, ref3)
    right_dist = padder.get_distance_to_ref_mask(dir='right')
    correct = [0, 1, 0, 0, 0]
    assert np.all(right_dist == correct)

    left_dist = padder.get_distance_to_ref_mask(dir='left')
    correct = [0, 0, 1, 0, 0]
    assert np.all(left_dist == correct)


def test_get_padded_variants_from_vcf():
    variants = get_padded_variants_from_vcf("../example_data/few_variants_two_chromosomes.vcf", "../example_data/small_reference_two_chromosomes.fa")
    print(variants)



def test_variant_padder_with_indel_after_snp_same_pos():
    variants = Variants.from_entry_tuples([
        ("chr1", 4, "ACGT", ""),
        ("chr1", 3, "A", "T"),
        ("chr1", 9, "A", "T")
    ])
    ref = bnp.as_encoded_array("AAAAACGTAAA")
    padder = VariantPadder(variants, ref)
    padded = padder.run()
    print(padded)

    assert padded == variants