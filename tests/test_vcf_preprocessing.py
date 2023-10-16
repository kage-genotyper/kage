from kage.benchmarking.vcf_preprocessing import preprocess_sv_vcf, get_cn0_ref_alt_sequences_from_vcf, find_snps_indels_covered_by_svs
import pytest
import bionumpy as bnp
import numpy as np

@pytest.fixture
def genome():
    return bnp.genomic_data.GenomicSequence.from_dict(
        {"chr1": "TACGGCAAAATTTACCCCCCCCCC"}
    )


@pytest.fixture
def variants():
    return bnp.datatypes.VCFEntry.from_entry_tuples(
        [
            ("chr1", 2, ".", "A", "<CN123>", "0", "PASS", "SVTYPE=DEL;END=200"),
            ("chr1", 5, ".", "C", "<CN0>", "0", "PASS", "SVTYPE=DEL;END=8"),
            ("chr1", 20, ".", "A", "<CN123>", "0", "PASS", "SVTYPE=DEL;END=200"),
        ]
    )


# mocking of genome not working with bionumpy
@pytest.mark.skip
def test_get_cn0_sequences(genome, variants):
    variants_cn0 = variants[1:2]

    ref, alt = get_cn0_ref_alt_sequences_from_vcf(variants_cn0, genome)

    assert ref[0].to_string() == "CAA"
    assert alt[0].to_string() == "C"

    print(ref, alt)


@pytest.fixture
def variants2():
    return bnp.datatypes.VCFEntry.from_entry_tuples(
        [
            ("chr1", 1, ".", "A", "T", ".", ".", "."),
            ("chr1", 2, ".", "A", "G", ".", ".", "."),
            ("chr1", 2, ".", "ACACACAC", "A", ".", ".", "."),
            ("chr1", 5, ".", "C", "G", ".", ".", "."),
            ("chr1", 9, ".", "AT", "A", ".", ".", "."),
            ("chr1", 8, ".", "AT", "A", ".", ".", "."),
        ]
    )


def test_find_snps_indels_covered_by_svs(variants2):
    is_covered = find_snps_indels_covered_by_svs(variants2, sv_size_limit=3)
    assert np.all(is_covered == [False, False, False, True, False, True])

