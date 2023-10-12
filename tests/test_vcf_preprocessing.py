from kage.benchmarking.vcf_preprocessing import preprocess_sv_vcf, get_cn0_ref_alt_sequences_from_vcf
import pytest
import bionumpy as bnp


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


def test_get_cn0_sequences(genome, variants):
    variants_cn0 = variants[1:2]

    ref, alt = get_cn0_ref_alt_sequences_from_vcf(variants_cn0, genome)

    assert ref[0].to_string() == "CAA"
    assert alt[0].to_string() == "C"

    print(ref, alt)
