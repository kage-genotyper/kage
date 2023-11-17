#from kage.preproessing.variants import MultiAllelelicVariants
import npstructures
import pytest
from bionumpy import Interval
from kage.indexing.graph import Graph, GenomeBetweenVariants
from kage.indexing.kmer_scoring import make_kmer_scorer_from_random_haplotypes
from kage.indexing.path_variant_indexing import find_tricky_variants_from_multiallelic_signatures
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.preprocessing.variants import MultiAllelicVariantSequences, Variants, \
    VariantAlleleToNodeMap, VariantPadder
from kage.indexing.paths import PathCreator, PathSequences
import bionumpy as bnp
from kage.indexing.signatures import MatrixVariantWindowKmers, MultiAllelicSignatures, MultiAllelicSignatureFinderV2, \
    VariantWindowKmers2, get_signatures
import numpy as np
from kage.indexing.paths import PathCombinationMatrix, Paths
from kage.indexing.signatures import MultiAllelicSignatureFinder
import awkward as ak
from kage.indexing.graph import make_multiallelic_graph
from graph_kmer_index import sequence_to_kmer_hash
from kage.indexing.path_based_count_model import PathBasedMappingModelCreator

@pytest.fixture
def variant_alleles():
    return MultiAllelicVariantSequences.from_list([
        ["A", "C"],
        ["ACTG", "ATTG", ""]
    ])

@pytest.fixture
def genome_between_variants():
    return GenomeBetweenVariants.from_list(["AAAA", "GGGG", "TTTT"])


@pytest.fixture
def graph(genome_between_variants, variant_alleles):
    return Graph(genome_between_variants, variant_alleles)


def test_multi_alleles(variant_alleles):
    sequences = variant_alleles.get_haplotype_sequence([0, 0])
    assert sequences.tolist() == ["A", "ACTG"]
    sequences = variant_alleles.get_haplotype_sequence([1, 2])
    assert sequences.tolist() == ["C", ""]


def test_graph(graph):
    assert graph.sequence([1, 2]).tolist() == ["AAAA", "C", "GGGG", "", "TTTT"]
    assert graph.sequence([0, 0]).tolist() == ["AAAA", "A", "GGGG", "ACTG", "TTTT"]


class DummyScorer2:
    def score_kmers(self, kmers):
        return np.zeros_like(kmers)

    def __getitem__(self, item):
        return self.score_kmers(item)


def test_multiallelic_signature_finder():
    paths = Paths(
        PathSequences.from_list([
            ["AAA", "CCC", "G", "TT", "AAA"],
            ["AAA", "CCC", "G", "TT", "AAA"],
            ["AAA", "G", "G", "CC", "AAA"],
        ]),
        PathCombinationMatrix([
            [0, 1],
            [0, 1],
            [1, 0],
            ]
        )
    )

    signature_finder = MultiAllelicSignatureFinder(paths, k=3, scorer=DummyScorer2())
    signatures = signature_finder.run()

    s = ak.to_list(signatures.signatures)
    assert len(s[0][0]) == 1
    assert len(s[0][1]) == 1
    assert len(s[1][0]) == 1
    assert len(s[1][1]) == 1


def test_multiallelic_signature_finder_on_three_following_snps():
    # testing all the way from ref + variants
    window = 4
    k = 6
    reference_sequences = bnp.datatypes.SequenceEntry.from_entry_tuples([
        ("chr1", "CCCC" "ACTG" "GGGGAGGGG")
    ])
    variants = Variants.from_entry_tuples([
        ("chr1", 4, "A", "C"),
        ("chr1", 5, "C", "G"),
        ("chr1", 6, "T", "A"),
        ("chr1", 7, "G", "A"),
    ])
    graph, node_mapping = make_multiallelic_graph(reference_sequences, variants)
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    paths = PathCreator(graph,
                        window=window,  # bigger windows to get more paths when multiallelic
                        make_disc_backed=False,
                        disc_backed_file_base_name="test.tmp").run(n_alleles_per_variant)

    kmers_to_avoid = ["tagggg", "ctgggg", "aagggg"]
    kmers_to_avoid = [sequence_to_kmer_hash(kmer) for kmer in kmers_to_avoid]

    class Scorer(DummyScorer2):
        def score_kmers(self, kmers):
            return np.array([-100 if kmer in kmers_to_avoid else 0 for kmer in kmers])

    signatures = get_signatures(k, paths, Scorer(), chunk_size=1)
    s = signatures.to_list_of_sequences(k)

    # variant 3 should have at least 4 kmers if things are correct
    assert len(s[2][0]) >= 4
    assert len(s[2][1]) >= 4


def test_multiallelic_signature_finder_special_case():
    # case that seems weird on real data
    window = 6
    k = 31
    reference_sequences = bnp.datatypes.SequenceEntry.from_entry_tuples([
        ("chr1", "ccattcgag" "tccagtccat" "tccattcc" "a" "a" "t" "acatttcgttccattccattccattccatt" "c"*30)
    ])
    variants = Variants.from_entry_tuples([
        ("chr1", 9, "", "TCCAGTCCAT"),
        ("chr1", 27, "A", "T"),
        ("chr1", 28, "A", "G"),
        ("chr1", 29, "T", "A"),
        ("chr1", 40, "C", "A"),
    ])
    graph, node_mapping = make_multiallelic_graph(reference_sequences, variants)
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    paths = PathCreator(graph,
                        window=window,  # bigger windows to get more paths when multiallelic
                        make_disc_backed=False,
                        disc_backed_file_base_name="test.tmp").run(n_alleles_per_variant)

    signatures = get_signatures(k, paths, scorer=DummyScorer2(), add_dummy_count_to_index=1)
    signatures.describe(k)
    s = signatures.to_list_of_sequences(k)
    # allele at variant 4 should have 2**3 kmers since all combinations of previous variants should be included
    assert len(s[3][0]) == 8
    assert len(s[3][1]) == 8
    assert "ccagtccattccagtccattccattcctata" in s[3][0]


def test_that_ref_sequence_is_always_in_signature():
    k = 31
    window = 5
    reference_sequences = bnp.datatypes.SequenceEntry.from_entry_tuples([
        ("chr1", "T" * 100)
    ])
    variants = Variants.from_entry_tuples([
        ("chr1", i, "T", "A") for i in range(10, 50, 4)
    ])


    graph, node_mapping = make_multiallelic_graph(reference_sequences, variants)
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    paths = PathCreator(graph,
                        window=window,  # bigger windows to get more paths when multiallelic
                        make_disc_backed=False,
                        disc_backed_file_base_name="test.tmp").run(n_alleles_per_variant)

    signatures = get_signatures(k, paths, scorer=DummyScorer2(), add_dummy_count_to_index=1)
    signatures.describe(k)
    s = signatures.to_list_of_sequences(k)

    for variant_id in range(len(variants)):
        assert "t"*31 in s[variant_id][0]

@pytest.fixture
def bnp_reference_sequences():
    reference_sequences = bnp.datatypes.SequenceEntry.from_entry_tuples([
        ("chr1", "CCTG" + "ACTG" * 2 + "CCCC" * 2)
    ])
    return reference_sequences


@pytest.fixture
def bnp_variants():
    variants = Variants.from_entry_tuples([
        ("chr1", 3, "G", "T"),
        ("chr1", 7, "G", "T"),
        ("chr1", 7, "G", "A"),
        ("chr1", 8, "ACTG", ""),
        ("chr1", 16, "", "GG")
    ])
    return variants


def test_integration_from_variants_to_signatures(bnp_reference_sequences, bnp_variants):
    k = 4
    window = 3
    graph, node_mapping = make_multiallelic_graph(bnp_reference_sequences, bnp_variants)
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    paths = PathCreator(graph,
                        window=window,  # bigger windows to get more paths when multiallelic
                        make_disc_backed=False,
                        disc_backed_file_base_name="test.tmp"
                        ).run(n_alleles_per_variant)

    print(paths.variant_alleles)

    variant_window_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(paths.paths, k)
    variant_window_kmers = VariantWindowKmers2.from_matrix_variant_window_kmers(variant_window_kmers, paths.variant_alleles.matrix)
    variant_window_kmers.describe(k)

    # scorer = DummyScorer2()
    # haplotype matrix
    biallelic_haplotype_matrix = SparseHaplotypeMatrix.from_variants_and_haplotypes(
        np.array([0, 1, 2, 3, 4,
                  0, 1, 2, 3, 4,
                  0, 1, 2, 3, 4,
                  0, 1, 2, 3, 4]),
        np.array([1, 1, 1, 1, 1,
                  2, 3, 2, 3, 2,
                  6, 6, 6, 6, 6,
                  7, 7, 7, 7, 7]),
        n_variants=5,
        n_haplotypes=8
    )

    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    haplotype_matrix = biallelic_haplotype_matrix.to_multiallelic(n_alleles_per_variant)

    scorer = make_kmer_scorer_from_random_haplotypes(graph, haplotype_matrix, k, n_haplotypes=8, modulo=2000003)
    assert scorer.score_kmers(sequence_to_kmer_hash("cgcg")) == 0  # does not exist in graph
    assert scorer.score_kmers(sequence_to_kmer_hash("cctg")) < 0  # should exist, score is negative frequency
    assert scorer.score_kmers(sequence_to_kmer_hash("cctg")) > scorer.score_kmers(sequence_to_kmer_hash("actg"))  # cctg less frequent than actg
    assert scorer.score_kmers(sequence_to_kmer_hash("acta") > scorer.score_kmers(sequence_to_kmer_hash("aact"))) - scorer.score_kmers(sequence_to_kmer_hash("accc"))

    signatures = MultiAllelicSignatureFinderV2(variant_window_kmers, scorer=scorer, k=k).run()
    signatures.describe(k)
    s = signatures.to_list_of_sequences(k)

    # least frequenct kmer should be chosen, last kmer on a tie
    assert s[0][0] == ["cctg"]  # cctg should be preferred over gact because lower freq
    assert s[0][1] == ["tact"]  # all have same freq, tact is the last

    # variant 2 (multiallelic)
    assert s[1][1] == ["actt"]

    # variant 3 (deletion)
    #assert set(s[2][0]) == set(["ctgc"])
    assert "gccc" not in s[2][1]

    # variant 4 (insertion)
    assert s[3][0] == ["cccc"]
    assert "gccc" not in s[3][1]

    # check parts of kmer index
    kmer_index = signatures.get_as_kmer_index(node_mapping=node_mapping, modulo=123, k=k)
    assert node_mapping.get_alt_node(0) in kmer_index.get_nodes(sequence_to_kmer_hash("tact"))
    assert node_mapping.get_alt_node(1) in kmer_index.get_nodes(sequence_to_kmer_hash("actt"))

    print(biallelic_haplotype_matrix.to_matrix())
    print(haplotype_matrix.to_matrix())


    model_creator = PathBasedMappingModelCreator(graph, kmer_index,
                                                 haplotype_matrix,
                                                 k=k,
                                                 paths_allele_matrix=paths.variant_alleles,
                                                 window=window-1,  # use lower window for better matching of haplotypes to paths
                                                 max_count=20,
                                                 node_map=node_mapping,
                                                 n_nodes=5*2)
    count_model = model_creator.run()

    for variant_id in range(5):
        ref_node = node_mapping.get_ref_node(variant_id)
        var_node = node_mapping.get_alt_node(variant_id)
        print("Variant ", variant_id, "nodes:", ref_node, var_node)
        print(count_model.describe_node(ref_node))
        print(count_model.describe_node(var_node))


    # check tricky variants, variant ID 3 should be tricky because shared kmers
    tricky_variants = find_tricky_variants_from_multiallelic_signatures(signatures,
                                                                        node_mapping.n_biallelic_variants)

    # shared kmers not tricky anymore
    #assert tricky_variants.tricky_variants[3] == True


def test_that_fewer_signatures_are_preferred_integration():
    # check that window giving fewer signatures are preferred
    # when sum of scores on a window is used for scoring
    pass


@pytest.fixture
def paths():
    return Paths(
        PathSequences.from_list([
            ["AAA", "CCC", "G", "TT", "AAA"],
            ["AAA", "CCC", "G", "TT", "AAA"],
            ["AAA", "G", "G", "CC", "AAA"],
        ]),
        PathCombinationMatrix([
            [0, 1],
            [0, 1],
            [1, 0],
            ]
        )
    )


def test_multiallelic_signature_finder2(paths):
    signature_finder = MultiAllelicSignatureFinder(paths, k=3, scorer=DummyScorer2())
    signatures = signature_finder.run()

    s = ak.to_list(signatures.signatures)
    assert len(s[0][0]) == 1
    assert len(s[0][1]) == 1


def test_multiallelic_signatures_as_kmer_index():
    signatures = MultiAllelicSignatures.from_list([
        [[1], [10, 11], [20]],
        [[5, 6], [5]]
    ])
    node_mapping = VariantAlleleToNodeMap(
        npstructures.RaggedArray([
            [0, 1, 2],
            [3, 4]
        ]),
        biallelic_ref_nodes=np.array([0, 0, 3]),
        biallelic_alt_nodes=np.array([1, 2, 4])
    )

    index = signatures.get_as_kmer_index(3, node_mapping)
    assert np.all(index.get_nodes(1) == [0])
    assert np.all(index.get_nodes(10) == [1])
    assert np.all(index.get_nodes(11) == [1])
    assert np.all(index.get_nodes(5) == [3, 4])


def test_multialellic_signature_finderv2(paths):
    k = 3
    #old = MultiAllelicSignatureFinder(paths, scorer=DummyScorer2(), k=k).run()
    #print(old.describe(k))

    variant_window_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(paths.paths, k)
    variant_window_kmers = VariantWindowKmers2.from_matrix_variant_window_kmers(variant_window_kmers, paths.variant_alleles.matrix)
    new = MultiAllelicSignatureFinderV2(variant_window_kmers, scorer=DummyScorer2(), k=k).run()

    print(new.describe(k))
    s = new.to_list_of_sequences(k)

    # first variant, will pick last kmer in window
    assert s[0][0] == ["cgt"]
    assert s[0][1] == ["ggc"]

    # second variant
    assert s[1][0] == ["caa"]
    assert s[1][1] == ["taa"]

    #assert np.all(old.signatures == new.signatures)


def test_signatures_on_graph_with_many_alleles_integration():
    window = 5
    k = 3
    variants = Variants.from_entry_tuples(
        [
            # many variants which will be merged
            ("chr1", 5, "ACTG", ""),
            ("chr1", 5, "A", "C"),
            ("chr1", 5, "A", "T"),
            ("chr1", 5, "A", "G"),
            ("chr1", 6, "C", "A"),
            ("chr1", 6, "C", "T"),
            ("chr1", 6, "C", "G"),
            ("chr1", 7, "T", "G"),
            ("chr1", 7, "T", "A"),
            ("chr1", 7, "T", "C"),
            # not overlapping
            ("chr1", 10, "G", "T"),
            ("chr1", 11, "G", "T"),
            ("chr1", 12, "G", "T"),
            ("chr1", 13, "G", "T"),
        ]
    )

    reference = bnp.datatypes.SequenceEntry.from_entry_tuples([("chr1", "CCCCACTGGGGGGGGGGGGG")])
    padder = VariantPadder(variants, reference.sequence[0])
    padded_variants = padder.run()
    print(padded_variants)

    graph, node_mapping = make_multiallelic_graph(reference, padded_variants)
    n_alleles_per_variant = node_mapping.n_alleles_per_variant

    paths = PathCreator(graph,
                        window=window,  # bigger windows to get more paths when multiallelic
                        make_disc_backed=False,
                        disc_backed_file_base_name="test.tmp").run(n_alleles_per_variant)

    variant_window_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(paths.paths, k)
    variant_window_kmers = VariantWindowKmers2.from_matrix_variant_window_kmers(variant_window_kmers,
                                                                                paths.variant_alleles.matrix)
    variant_window_kmers.describe(k)
    kmers = variant_window_kmers.kmers

    # first kmer in allele 0 variant 0 should be common on alle paths
    k = kmers[0, 0, :, 0]
    print(k)
    assert len(np.unique(kmers[0, 0, :, 0])) == 1


@pytest.fixture
def variants():
    return Variants.from_entry_tuples([
        ("1", 1, "A", "C"),
        ("1", 10, "ACTG", "ATTG"),
        ("1", 10, "ACTG", ""),
        ("2", 10, "A", "C")
    ])


@pytest.fixture
def reference_sequence():
    return bnp.datatypes.SequenceEntry.from_entry_tuples([
        ("1", "CACCCCCCCCACTGCCCC"),
        ("2", "GGGGGGGGGGAGGG")
    ])


def test_get_multiallelic_variant_alleles_from_variants(variants):
    multiallelic, node_ids, intervals = variants.get_multi_allele_variant_alleles()
    assert multiallelic.to_list() == [["A", "C"], ["ACTG", "ATTG", ""], ["A", "C"]]
    assert np.all(node_ids.lookup([0, 0], [0, 1]) == [0, 1])
    assert np.all(node_ids.lookup([1, 1, 1], [0, 1, 2]) == [2, 3, 4])
    assert np.all(node_ids.biallelic_ref_nodes == [0, 2, 2, 5])
    assert np.all(node_ids.biallelic_alt_nodes == [1, 3, 4, 6])

    correct_interval = Interval.from_entry_tuples([
        ("1", 1, 2),
        ("1", 10, 14),
        ("2", 10, 11)
    ])

    assert np.all(intervals == correct_interval)


def test_make_multiallelic_graph(reference_sequence, variants):
    graph, node_mapping = make_multiallelic_graph(reference_sequence, variants)
    assert graph.genome.to_list() == ["C", "CCCCCCCC", "CCCCGGGGGGGGGG", "GGG"]
    assert graph.variants.to_list() == [["A", "C"], ["ACTG", "ATTG", ""], ["A", "C"]]


def test_replace_nonunique_variant_window_kmers():
    kmers = VariantWindowKmers2.from_list(
        # variant
        [
        [
           # allele
           [
               # path
               [1, 2, 3, 4, 5],
               [1, 5, 4, 3, 2],
               [8, 5, 1, 1, 2],
           ],
            # allele 2
            [
               [100, 200],
               [101, 10],
               [10, 200]
           ]
        ],
        # variant 2
        [
            [
                [1, 2],
                [10, 12],
                [10, 2]
            ]
        ]
    ])

    correct = VariantWindowKmers2.from_list([
        # variant
        [
            # allele
            [
                # path
                [1, 2, 1, 1, 2],
                [0, 5, 3, 3, 0],
                [8, 0, 4, 4, 5],
            ],
            # allele 2
            [
                [10, 10],
                [100, 200],
                [101, 0]
            ]
        ],
        # variant 2
        [
            [
                [1, 2],
                [10, 0],
                [0, 12]
            ]
        ]
    ])

    kmers.replace_nonunique_kmers(0)

    assert kmers == correct


def test_window_kmers_from_paths_with_flexible_window_size_many_variants():
    # test that things don't break with many variants
    n_sequences = 50001  # must be odd number
    n_paths = 10
    path_sequences = [
        ["AACCTTAACCTTAACCTTAACCTTAACCTTAACCTT"] + ["ACTG"] * n_sequences + ["GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"]
        for i in range(n_paths)
    ]
    path_sequences = PathSequences.from_list(path_sequences)
    kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(path_sequences, k=31)

    for kmer in kmers.kmers[:, 0, 0]:
        assert kmer == path_sequences[0][0:31]





@pytest.fixture
def paths_with_sv():
    sv_sequence = "C" * 10 + "G" + "C" * 10
    return Paths(
        PathSequences.from_list([
            ["ATATATAT", sv_sequence, "GGGGG", "TT", "C" * 10],
            ["ATATATAT", sv_sequence, "GGGGG", "TT", "C" * 10],
            ["ATATATAT", "C", "GGGGG", "CC", "C" * 10],
        ]),
        PathCombinationMatrix([
            [0, 1],
            [0, 1],
            [1, 0],
        ]
        )
    )

def test_manually_process_svs_multiallelic_signature_finder(paths_with_sv):
    """
    Initial kmer finder will pick "CGGGGG" since all kmers are equally
    scored and that is the last kmers. Manually process will manage to find some other
    kmers that are unique between the alleles
    """
    paths = paths_with_sv
    k = 5
    variant_window_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(paths.paths, k)
    variant_window_kmers = VariantWindowKmers2.from_matrix_variant_window_kmers(variant_window_kmers,
                                                                                paths.variant_alleles.matrix)

    # high sv min size
    signatures = MultiAllelicSignatureFinderV2(variant_window_kmers, scorer=DummyScorer2(), k=k, sv_min_size=1000000).run()
    print(signatures.signatures.type)
    s = signatures.to_list_of_sequences(k)
    assert s[0][0] == ["cgggg"]
    assert s[0][1] == ["cgggg"]

    print(signatures.describe(k))

    # low sv min size will capture sv
    signatures = MultiAllelicSignatureFinderV2(variant_window_kmers, scorer=DummyScorer2(), k=k,
                                               sv_min_size=7).run()
    print(signatures.signatures.type)
    s = signatures.to_list_of_sequences(k)
    assert "cgggg" not in s[0][0]
    assert "cgggg" not in s[0][1]


    print(signatures.describe(k))

    # first variant, will pick last kmer in window
    """
    assert s[0][0] == ["cgt"]
    assert s[0][1] == ["ggc"]

    # second variant
    assert s[1][0] == ["caa"]
    assert s[1][1] == ["taa"]
    """

    # assert np.all(old.signatures == new.signatures)


if __name__ == "__main__":
    test_integration_from_variants_to_signatures(bnp_reference_sequences(), bnp_variants())
