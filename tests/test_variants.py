import logging
logging.basicConfig(level=logging.INFO)
import pytest
import numpy as np
import bionumpy as bnp
from kage.preprocessing.variants import Variants, VariantStream, FilteredOnMaxAllelesVariantStream2, \
    VariantAlleleToNodeMap, filter_variants_with_more_alleles_than
from kage.io import SimpleVcfEntry
import npstructures as nps

def test_variants_from_multialellic_vcf():
    vcf = bnp.open("multiallelic.vcf").read()
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


def test_variants_to_simple_vcf_entry():
    ref = [
        ">chr1",
        "ACGTACAATTTTT",
        ">chr2",
        "ACGTACGT"
    ]
    with open("tmp.fasta", "w") as f:
        f.writelines([line + "\n" for line in ref])
        
    fasta = bnp.open_indexed("tmp.fasta")
    
    variants = Variants.from_entry_tuples([
        ("chr1", 4, "A", "T"),
        ("chr1", 6, "AA", ""),
        ("chr2", 3, "", "ACGT"),
    ])
    
    padded = variants.to_simple_vcf_entry_with_padded_indels(fasta)

    correct = SimpleVcfEntry.from_entry_tuples([
        ("chr1", 4, "A", "T"),
        ("chr1", 5, "CAA", "C"),
        ("chr2", 2, "G", "GACGT")
    ])

    assert np.all(padded == correct)


def test_variant_stream():
    variant_stream = VariantStream.from_vcf("multiallelic.vcf")
    for chunk in variant_stream.read_chunks():
        print(chunk)


    variant_stream = VariantStream.from_vcf("multiallelic.vcf")
    for chunk in variant_stream.read_by_chromosome():
        print(chunk)


def test_variant_filter_max_alleles():
    variant_stream = VariantStream.from_vcf("multiallelic2.vcf")
    filtered_stream = FilteredOnMaxAllelesVariantStream2(variant_stream, max_alleles=1)
    for chunk in filtered_stream.read_chunks():
        print(chunk)
        assert len(chunk) == 2


def test_find_variants_with_more_alleles_than():
    biallelic_variants = Variants.from_entry_tuples(
        [
            ("chr1", 4, "A", "T"),
            ("chr1", 6, "ATG", "T"),
            ("chr1", 6, "ATG", "G"),
            ("chr1", 6, "ATG", "C"),
            ("chr2", 8, "A", "C"),
            ("chr2", 8, "A", "G"),
        ]
    )
    original_variants = Variants.from_entry_tuples(
        [
            ("chr1", 4, "A", "T"),
            ("chr1", 6, "ATG", "T,G,C"),
            ("chr2", 8, "A", "C,G"),
        ]
    )
    n_alleles_per_original_variant = np.array([2, 4, 3])

    new_biallelic, new_original, new_n_alleles, filter = filter_variants_with_more_alleles_than(biallelic_variants, original_variants,
                                                    n_alleles_per_original_variant, 2)

    assert np.all(filter == [False, True, False])
    assert len(new_original) == 2
    assert len(new_biallelic) == 3
    assert np.all(new_n_alleles == [2, 3])

    print(filter)

