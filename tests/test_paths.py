import numpy as np
import pytest
from kage.indexing.paths import PathCreator, PathSequences, Paths
import bionumpy as bnp
from kage.util import n_unique_values_per_column


def test_multiallelic_paths():
    alleles = [2, 3, 2, 4]
    matrix = PathCreator.make_combination_matrix_multi_allele(alleles, window=3)
    print(matrix.matrix)
    assert len(matrix) == 2**3
    assert matrix.shape[1] == 4
    # All alleles should be represented
    for i, allele in enumerate(alleles):
        assert all(a in matrix.matrix[:, i] for a in range(allele))


def test_convert_biallelic_path_to_multiallelic():
    n_alleles_per_variant = [2, 2, 3, 2, 4, 2]
    #       #  #  #...  #  #......  #
    path = [0, 1, 0, 1, 1, 1, 1, 1, 1]
    converted = PathCreator.convert_biallelic_path_to_multiallelic(n_alleles_per_variant, path, how="path")

    correct = [0, 1, 2, 1, 3, 1]
    assert np.all(converted == correct)


@pytest.fixture
def path_sequences():
    return PathSequences.from_list([
        ["ACGT", "A", "GGG", "C", "TTT", "A", "CC"],
        ["ACGT", "G", "GGG", "A", "TTT", "G", "CC"],
    ])


def test_subset_path_sequence(path_sequences):

    subset = path_sequences.subset_on_variants(0, 1, padding=4)
    assert subset.sequences == PathSequences.from_list([
        ["ACGT", "A", "GGGC"],
        ["ACGT", "G", "GGGA"]
    ]).sequences

    subset = path_sequences.subset_on_variants(0, 1, padding=1)
    assert subset.sequences == PathSequences.from_list([
        ["T", "A", "G"],
        ["T", "G", "G"]
    ]).sequences

    subset = path_sequences.subset_on_variants(0, 2, padding=10)
    assert subset.sequences == PathSequences.from_list([
        ["ACGT", "A", "GGG", "C", "TTTACC"],
        ["ACGT", "G", "GGG", "A", "TTTGCC"]
    ]).sequences

    disc_backed = path_sequences.to_disc_backed("testpath")
    subset = disc_backed.subset_on_variants(0, 1, padding=4)

    sequences = list(subset.iter_path_sequences())
    assert np.all(sequences[0].sequence == bnp.as_encoded_array(["ACGT", "A", "GGGC"]))
    assert np.all(sequences[1].sequence == bnp.as_encoded_array(["ACGT", "G", "GGGA"]))


def test_subset_paths(path_sequences):
    paths = Paths(path_sequences,
                  np.array([
                      [0, 1, 0],
                      [1, 0, 0]
                  ])
    )

    subset = paths.subset_on_variants(1, 2, 4)
    assert np.all(subset.variant_alleles.matrix == [
        [1],
        [0]
    ])

    assert subset.paths.sequences == PathSequences.from_list([
        ["AGGG", "C", "TTTA"],
        ["AGGG", "A", "TTTG"]
    ]).sequences


def test_chunk_paths(path_sequences):
    paths = Paths(path_sequences,
                  np.array([
                      [0, 1, 0],
                      [1, 0, 0]
                  ])
    )
    paths.to_disc_backend("testpath")
    print(paths.paths)
    print(paths)
    chunked = paths.chunk([(0, 1), (1, 3)], padding=4)
    print(chunked)
    assert len(chunked) == 2
    assert np.all(chunked[0].variant_alleles.matrix == [[0], [1]])

    #assert chunked[0].paths.sequences[0].load() == path_sequences[0].subset_on_variants(0, 1, padding=4)


def test_path_creator_many_alleles_few_paths():
    # test that all alleles get covered
    n_alleles_per_variant = np.array(np.random.randint(2, 8, 50))
    print(n_alleles_per_variant)
    combination_matrix = PathCreator.make_combination_matrix_multi_allele_v2(n_alleles_per_variant, window=3)
    assert np.all(n_unique_values_per_column(combination_matrix.matrix) == n_alleles_per_variant)
    print(combination_matrix)


def test_path_combination_matrix_v2():
    n_alleles_per_variant = np.array([2, 3, 2, 4, 2])
    matrix = PathCreator.make_combination_matrix_multi_allele_v2(n_alleles_per_variant, window=3)
    assert np.all(n_unique_values_per_column(matrix) == n_alleles_per_variant)


    n_alleles_per_variant = np.array([2, 2, 2, 2, 2])
    matrix = PathCreator.make_combination_matrix_multi_allele_v2(n_alleles_per_variant, window=3)
    print(matrix)


def test_path_combination_matrix_v3():
    n_alleles_per_variant = np.array([2, 3, 2, 4, 2])
    matrix = PathCreator.make_combination_matrix_multi_allele_v3(n_alleles_per_variant, window=3)
    assert np.all(n_unique_values_per_column(matrix) == n_alleles_per_variant)
    print(matrix)
