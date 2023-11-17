import awkward as ak
import numpy as np
from kage.indexing.paths import PathSequences, DiscBackedPathSequence
from kage.indexing.signatures import MatrixVariantWindowKmers, VariantWindowKmers2
import bionumpy as bnp

from kage.util import log_memory_usage_now, get_memory_usage
import pytest

@pytest.fixture
def path_sequences():
    return PathSequences([
        bnp.as_encoded_array(["AAA", "CCC", "AAA", "GGG", "AAA"], bnp.DNAEncoding),
        bnp.as_encoded_array(["AAA", "", "CCC", "T", "AAA"], bnp.DNAEncoding),
    ])


def test_matrix_window_kmers_flexible_window_size(path_sequences):

    path_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(path_sequences, k=3)

    kmer_encoding = bnp.encodings.kmer_encodings.KmerEncoding(bnp.DNAEncoding, 3)
    kmers = [[[kmer_encoding.to_string(k) for k in kmer] for kmer in path] for path in path_kmers.kmers]

    correct = [
        [["AAC", "ACC", "CCC", "CCA", "CAA"], ["AAG", "AGG", "GGG", "GGA", "GAA"]],
        [["AAC", "ACC"], ["CCT", "CTA", "TAA"]]
    ]

    assert kmers == correct

    variant_alleles = np.array([
        [1, 0],
        [0, 1]
    ])
    assert np.all(path_kmers.get_kmers(0, 1, variant_alleles) == path_kmers.kmers[0][0])


def test_matrix_window_kmers_flexible_window_size_and_minimum_overlap_with_variant(path_sequences):
    # When minimum overlap with variants is 2:
    path_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(path_sequences, k=3, minimum_overlap_with_variant=2)
    kmer_encoding = bnp.encodings.kmer_encodings.KmerEncoding(bnp.DNAEncoding, 3)
    kmers = [[[kmer_encoding.to_string(k) for k in kmer] for kmer in path] for path in path_kmers.kmers]

    correct = [
        [["ACC", "CCC", "CCA", "CAA"], ["AGG", "GGG", "GGA", "GAA"]],
        [["ACC"], ["CTA", "TAA"]]
    ]

    assert kmers == correct



def test_variant_window_kmers2_from_matrix_variant_window_kmers_many_alleles_on_same_variant():
    n_alleles_on_variant = 10

    some_path = [
                # variant 1
                [0, 1, 2, 3],
                # variant 2
                [4, 5, 6, 7],
                # variant 3
                [1, 2]
            ]
    kmers = [some_path] * n_alleles_on_variant + \
            [
                # a different path
                [
                    [0, 1, 2, 3],
                    [10, 12],
                    [1, 2]
                ]
            ]
    kmers = MatrixVariantWindowKmers(
        ak.Array(kmers)
    )

    path_alleles_some_path = [0, 1, 0]
    path_alleles = np.array(
        [path_alleles_some_path] * n_alleles_on_variant +
        [[0, 0, 0]]
    )
    kmers2 = VariantWindowKmers2.from_matrix_variant_window_kmers(kmers, path_alleles)
    print(kmers2.describe(5))


# not fixed, but not critical
@pytest.mark.xfail
def test_variant_window_kmers2_from_matrix_variant_window_kmers_sparse_paths():
    """
    Some alleles are not covered by paths.
    This should still work and not crash
    """
    path_alleles = np.array([
        [0, 0, 0],
        [0, 2, 0],  # allele 1 is not covered
        [1, 2, 1]
    ])
    kmers = MatrixVariantWindowKmers(
        ak.Array([
            # path 1
            [
                # variant
                [0, 1, 2],
                # variant
                [1, 2, 3],
                [1, 2]
            ],
            # path 2
            [
                [1],
                [10],
                [20]
            ],
            [
                [5],
                [6],
                [7]
            ]
        ])
    )
    kmers2 = VariantWindowKmers2.from_matrix_variant_window_kmers(kmers, path_alleles)
    print(kmers2)


def test_benchmark_variant_window_kmers():
    n_variants = 10000
    n_paths = 64
    part = ["ACGT" * 10] + ["C"]
    sequences = []
    for i in range(n_paths):
        path_sequence = bnp.as_encoded_array(part * n_variants + ["ACGT" * 10], bnp.DNAEncoding)
        sequence = DiscBackedPathSequence.from_non_disc_backed(path_sequence, f"tmp_path_{i}")
        sequences.append(sequence)
    print("Memory start", get_memory_usage())

    path_alleles = np.random.randint(0, 2, size=(n_paths, n_variants))
    path_sequences = PathSequences(sequences)

    print("Memory: ", get_memory_usage())
    path_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(path_sequences, k=31)
    print("Memory: ", get_memory_usage())

    kmers2 = VariantWindowKmers2.from_matrix_variant_window_kmers(path_kmers, path_alleles)
    print("Memory: ", get_memory_usage())


    print(path_kmers.kmers)
    print(kmers2)


def test_find_kmers_only_inside_big_alleles():
    paths = PathSequences.from_list(
        [
            ["AAAA", "TGGGGGGGGGGGC", "AAAA", "CC", "ACTG"],
            ["ACTG", "GG", "ACTG", "CC", "ACTG"],
        ]
    )

    kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(paths, k=3, only_pick_kmers_inside_big_alleles=True)

    path_alleles = np.array([
        [0, 0],
        [1, 1]
    ])

    signatures = VariantWindowKmers2.from_matrix_variant_window_kmers(kmers, path_alleles)

    string_kmers = signatures.to_kmer_list(k=3)

    assert string_kmers[0][0][0][0] == "tgg", "First kmer on first allele of first variant should be beginning of first allele"
    assert string_kmers[0][0][0][-1] == "ggc"  # last kmer
    assert string_kmers[0][1][0][0] == "tgg"  # short alleles should have kmer starting before allele
    print(string_kmers)
