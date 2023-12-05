import numpy as np
import pytest
from kage.indexing.graph import Graph, GenomeBetweenVariants
from kage.indexing.paths import PathCreator, PathSequences, Paths, PathCombinationMatrix
import bionumpy as bnp
from kage.preprocessing.variants import MultiAllelicVariantSequences
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
    assert np.all(sequences[0].get_sequence() == bnp.as_encoded_array(["ACGT", "A", "GGGC"]))
    assert np.all(sequences[1].get_sequence() == bnp.as_encoded_array(["ACGT", "G", "GGGA"]))


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


def test_add_paths_with_missing_alleles_v2():
    n_alleles_per_variant = np.array([2, 3, 2, 4, 2])
    matrix = PathCombinationMatrix([
        [0, 0, 0, 3, 0],
        [0, 3, 1, 3, 1],
        [0, 0, 0, 3, 0],
        [0, 0, 1, 2, 0],
        [0, 0, 0, 2, 0],
    ])
    matrix.add_paths_with_missing_alleles_by_changing_existing_paths(n_alleles_per_variant)
    print(matrix.matrix)
    matrix.sanity_check()


@pytest.fixture
def graph():
    graph = Graph(
        genome=GenomeBetweenVariants.from_list(["ACTG", "AAA", "", "GGG", "CCCCCC"]),
        variants=MultiAllelicVariantSequences.from_list(
            [["A", "C"], ["A", "C", "T"], ["AA", "CC"], ["G", "T"]]
        )
    )
    return graph


def test_graph_backed_path_sequence(graph):
    paths = PathCreator(graph, window=2, make_graph_backed_sequences=True).run()
    path0_sequence = paths.paths[0].get_sequence().ravel().to_string()
    assert path0_sequence == "ACTG" + "A" + "AAA" + "A" + "" + "AA" + "GGG" + "G" + "CCCCCC"

    # subsetting
    subset = paths.paths[0].subset_on_variants(0, 3, padding=3)
    seq = subset.get_sequence()
    assert seq.tolist() == ["CTG", "A", "AAA", "A", "", "AA", "GGG"]

    subset = paths.paths[0].subset_on_variants(1, 3, padding=1)
    seq = subset.get_sequence()
    assert seq.tolist() == ["A", "A", "", "AA", "G"]


def test_sequence_after_variant_at_graph(graph):
    seq = graph.get_bases_after_variant(0, np.array([0, 0, 0, 0]), n_bases=4)
    assert seq.to_string() == "AAAA"

    seq = graph.get_bases_after_variant(0, np.array([0, 0, 0, 0]), n_bases=7)
    assert seq.to_string() == "AAAAAAG"

    seq = graph.get_bases_after_variant(0, np.array([0, 0, 0, 0]), n_bases=100)
    assert seq.to_string() == "AAAAAAGGGGCCCCCC"

    seq = graph.get_bases_after_variant(0, np.array([1, 2, 1, 1]), n_bases=100)
    assert seq.to_string() == "AAA" + "T" + "" + "CC" + "GGG" + "T" + "CCCCCC"

    seq = graph.get_bases_after_variant(1, np.array([1, 2, 1, 1]), n_bases=5)
    assert seq.to_string() == "" + "CC" + "GGG"


def test_sequence_before_variant_at_graph(graph):
    seq = graph.get_bases_before_variant(1, np.array([0, 0, 0, 0]), n_bases=100)
    assert seq.to_string() == "ACTG" + "A" + "AAA"

    seq = graph.get_bases_before_variant(1, np.array([0, 0, 0, 0]), n_bases=5)
    assert seq.to_string() == "G" + "A" + "AAA"

    seq = graph.get_bases_before_variant(3, np.array([0, 0, 0, 0]), n_bases=8)
    assert seq.to_string() == "AA" + "A" + "" + "AA" + "GGG"

    seq = graph.get_bases_before_variant(3, np.array([0, 2, 0, 0]), n_bases=8)
    assert seq.to_string() == "AA" + "T" + "" + "AA" + "GGG"


def test_get_graph_sequence_from_to_variant(graph):
    haplotype = np.array([0, 2, 1, 0])
    seq = graph.sequence(haplotype, from_to_variant=(0, 4))
    assert seq.tolist() == ["A", "AAA", "T", "", "CC", "GGG", "G"]

    seq = graph.sequence(haplotype, from_to_variant=(1, 4))
    assert seq.tolist() == ["T", "", "CC", "GGG", "G"]

    seq = graph.sequence(haplotype, from_to_variant=(1, 2))
    assert seq.tolist() == ["T"]

    seq = graph.sequence(haplotype, from_to_variant=(0, 3))
    assert seq.tolist() == ["A", "AAA", "T", "", "CC"]
