import pytest
from kage.indexing.path_variant_indexing import Paths, Variants, PathCreator, GenomeBetweenVariants
import bionumpy as bnp


@pytest.fixture
def variants():
    allele_sequences = [
        bnp.as_encoded_array(["A", "A", "", "A", "A"], bnp.DNAEncoding),  # ref
        bnp.as_encoded_array(["C", "C", "C", "C", "C"], bnp.DNAEncoding),  # alt
    ]
    return Variants(allele_sequences)


@pytest.fixture
def genome():
    return GenomeBetweenVariants(bnp.as_encoded_array(["GGG", "GG", "GG", "GG", "", "GGG"], bnp.DNAEncoding))


def test_create_path(genome, variants):
    print(genome.sequence)
    print(variants.allele_sequences)
    creator = PathCreator(variants, genome)
    paths = creator.run()
    print(paths)
    assert len(paths.paths) == 8


def test_get_kmers(genome, variants):
    creator = PathCreator(variants, genome)
    paths = creator.run()
    print(paths)
