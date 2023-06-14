#from kage.preprocessing.variants import MultiAllelelicVariants
import pytest
from kage.indexing.graph import Graph, VariantAlleleSequences, GenomeBetweenVariants, MultiAllelicVariantSequences


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

