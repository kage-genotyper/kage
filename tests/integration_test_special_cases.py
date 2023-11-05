import pytest
from kage.indexing.main import make_index
from kage.command_line_interface import genotype
from kage.util import make_args_for_genotype_command


def test_multiallelic_similar_alleles():
    vcf = "tricky_variants.vcf"
    ref = "ref_for_tricky_variants.fa"

    index, signatures = make_index(ref, vcf, "test_index.npz",
               k=31, modulo=2000033,
               variant_window=5,
               n_threads=4)

    signatures.describe(31)

    # Should manage to find unique kmers for all these alleles (tricky, but possible)
    sig = signatures.to_list_of_sequences(k=31)
    variant0_signatures = sig[0]
    assert all(len(s) == 1 for s in variant0_signatures), "Should be one kmer for each allele"
    kmers = [s[0] for s in variant0_signatures]
    assert len(set(kmers)) == len(kmers)

    args = make_args_for_genotype_command("test_index.npz", "reads.fq.gz", out_file="test_genotypes.vcf")
    genotype(args)


@pytest.mark.skip
def test2():
    vcf = "tricky_variants2.vcf"
    ref = "ref_for_tricky_variants2.fa"

    index, signatures = make_index(ref, vcf, "test_index.npz",
                                   k=31, modulo=2000033,
                                   variant_window=5,
                                   n_threads=4)

    signatures.describe(31)


@pytest.mark.skip
def test3():
    vcf = "tricky_variants4.vcf"
    ref = "21.fa"

    index, signatures = make_index(ref, vcf, "test_index.npz",
                                   k=31, modulo=2000033,
                                   variant_window=5,
                                   n_threads=4)

    signatures.describe(31)


def test5():
    """
    Small version of smaller svs that are overlapping inside a large complex sv
    Tricky/impossible to find unique kmer for the ref since all variants are extended.
    Should maybe try to mark ref allele as tricky allele and find good kmers for alt
    """
    vcf = "tricky_variants5.vcf"
    ref = "ref_for_tricky_variants5.fa"

    index, signatures = make_index(ref, vcf, "test_index.npz",
                                   k=31, modulo=2000033,
                                   variant_window=5,
                                   n_threads=4)

    signatures.describe(31)

if __name__ == "__main__":
   test5()
