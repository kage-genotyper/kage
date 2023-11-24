import numpy as np
import pytest
from graph_kmer_index import kmer_hash_to_sequence

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

    tricky_ref, tricky_alt = index.tricky_alleles
    assert np.sum(tricky_ref.tricky_variants) > 0
    assert np.sum(tricky_alt.tricky_variants) > 0
    print(tricky_ref)
    print(tricky_alt)


@pytest.mark.skip
def test6():
    vcf = "tricky_variants25.vcf"
    ref = "ref_for_tricky_variants25.fa"
    #ref = "21.fa"

    index, signatures, orig_count_model = make_index(ref, vcf, "test_index.npz",
                                   k=31, modulo=2000033,
                                   variant_window=7,
                                   n_threads=4)
    signatures.describe(31)

    biallelic_kmers = signatures.to_biallelic_list_of_sequences(k=31)

    tricky_ref, tricky_alt = index.tricky_alleles
    print(tricky_ref.tricky_variants)
    print(tricky_alt.tricky_variants)

    count_model_ref, count_model_alt = index.count_model
    print("Helper variants:")
    print(index.helper_variants.helper_variants)
    for variant_id, v in enumerate(index.vcf_variants):
        print("Variant", variant_id, v)
        print("\n".join(
            [", ".join(k) for k in biallelic_kmers[variant_id]]
        ))
        print(count_model_ref.describe_node(variant_id))
        print(count_model_alt.describe_node(variant_id))

        print("Original count model")
        ref_node = index.variant_to_nodes.ref_nodes[variant_id]
        alt_node = index.variant_to_nodes.var_nodes[variant_id]
        print(orig_count_model.describe_node(ref_node))
        print(orig_count_model.describe_node(alt_node))


def test_copy_number():
    # two variants with different number of a kmer
    vcf = "tricky_variants8.vcf"
    ref = "ref_for_tricky_variants8.fa"
    k = 4

    index, signatures, orig_count_model = make_index(ref, vcf, "test_index.npz",
                                   k=k, modulo=2000033,
                                   variant_window=7,
                                   n_threads=4)

    signatures.describe(k)

    print("Kmer index")
    print(index.kmer_index._kmers)
    print([kmer_hash_to_sequence(kmer, k) for kmer in index.kmer_index._kmers])
    print(index.kmer_index._nodes)

    count_model_ref, count_model_alt = index.count_model
    for variant_id in range(2):
        print("Variant", variant_id)
        print(count_model_ref.describe_node(variant_id))
        print(count_model_alt.describe_node(variant_id))

        print("Original count model")
        ref_node = index.variant_to_nodes.ref_nodes[variant_id]
        alt_node = index.variant_to_nodes.var_nodes[variant_id]
        print(orig_count_model.describe_node(ref_node))
        print(orig_count_model.describe_node(alt_node))

    args = make_args_for_genotype_command("test_index.npz",
                                          "reads_for_tricky_variants8.fa",
                                          out_file="test_genotypes.vcf",
                                          average_coverage=4.0,
                                          kmer_size=k)

    genotype(args)

    probs = np.load("test_genotypes.vcf.probs.npy")
    count_probs = np.load("test_genotypes.vcf.count_probs.npy")
    node_counts = np.load("test_genotypes.vcf.node_counts.npy")
    numeric_genotypes = np.load("test_genotypes.vcf.genotypes.npy")
    print(numeric_genotypes)

    print("PROBS")
    print(probs)
    print("Count probs")
    print(count_probs)

    print("Node counts")
    print(node_counts)

    assert np.all(numeric_genotypes == [2, 1])  # 1/1, 0/0



def test_copy_number():
    vcf = "tricky_variants13.vcf"
    # todo implement
    # variants have different frequencies of kmers
    # challenge is to find kmers that have different frequency on the two alleles,
    # which may not be kmers that have the lowest overall frequency
    pass

if __name__ == "__main__":
    test6()
    #test_copy_number()
