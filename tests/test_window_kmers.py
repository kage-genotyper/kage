import awkward as ak
import numpy as np
from kage.indexing.paths import PathSequences, DiscBackedPathSequence
from kage.indexing.signatures import MatrixVariantWindowKmers, VariantWindowKmers2
import bionumpy as bnp

from kage.util import log_memory_usage_now, get_memory_usage


def test_matrix_window_kmers_flexible_window_size():
    path_sequences = PathSequences([
        bnp.as_encoded_array(["AAA", "CCC", "AAA", "GGG", "AAA"], bnp.DNAEncoding),
        bnp.as_encoded_array(["AAA", "", "CCC", "T", "AAA"], bnp.DNAEncoding),
    ])
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
    print(kmers)
    kmers = MatrixVariantWindowKmers(
        ak.Array(kmers)
    )

    print(kmers.describe(5))

    path_alleles_some_path = [0, 1, 0]
    path_alleles = np.array(
        [path_alleles_some_path] * n_alleles_on_variant +
        [[0, 0, 0]]
    )

    kmers2 = VariantWindowKmers2.from_matrix_variant_window_kmers(kmers, path_alleles)

    print(kmers2.describe(5))


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