import itertools
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
import bionumpy as bnp
from bionumpy.bnpdataclass import bnpdataclass


"""
Module for simple variant signature finding by using static predetermined paths through the "graph".

Works when all variants are biallelic and there are no overlapping variants.

"""


@dataclass
class GenomeBetweenVariants:
    """ Represents the linear reference genome between variants, not including variant alleles"""
    sequence: bnp.EncodedRaggedArray


@dataclass
class Variants:
    allele_sequences: List[bnp.EncodedRaggedArray]  # one EncodedRaggedArray for each allele

    def n_variants(self):
        return len(self.allele_sequences[0])

@dataclass
class Paths:
    paths: List[bnp.EncodedRaggedArray]
    # the allele present at each variant in each path
    variant_alleles: np.ndarray
    #path_kmers: List[bnp.EncodedRaggedArray] = None

    def paths_for_allele_at_variant(self, allele, variant):
        relevant_paths = np.where(self.variant_alleles[:, variant] == allele)[0]
        return [self.paths[p] for p in relevant_paths]


    def get_kmers(self, variant, allele, kmer_size=3):
        # gets all kmers of size kmer_size around the variant allele (on all paths relevant for the allele)
        path_sequences = self.paths_for_allele_at_variant(allele, variant)
        window_sequences = []
        for path in path_sequences:
            offset_in_path = path._shape.starts[variant * 2 + 1]
            start = max(0, offset_in_path - kmer_size)
            end = offset_in_path + kmer_size
            window_sequences.append(path.ravel()[start:end].to_string())

        window_sequences = bnp.as_encoded_array(window_sequences, bnp.DNAEncoding)
        kmers = bnp.get_kmers(window_sequences, kmer_size)
        return kmers


class PathCreator:
    def __init__(self, variants: Variants, genome: GenomeBetweenVariants, window: int = 3):
        self._variants = variants
        self._genome = genome
        assert len(genome.sequence) == self._variants.n_variants() + 1
        self._window = window

    def run(self):
        alleles = [0, 1]  # possible alleles, assuming biallelic variants
        n_paths = len(alleles)**self._window
        n_variants = self._variants.n_variants()
        combinations = self._make_combination_matrix(alleles, n_paths, n_variants)

        paths = []

        # each path will have sequences between each variant and the allele sequences specified in combinations
        ref_between = self._genome.sequence
        for path, alleles in enumerate(combinations):
            # make a new EncodedRaggedArray where every other row is ref/variant
            n_rows = len(ref_between) + n_variants
            row_lengths = np.zeros(n_rows)
            row_lengths[0::2] = ref_between.shape[1]


            variant_sequences = bnp.as_encoded_array(
                [self._variants.allele_sequences[allele][variant].to_string() for variant, allele in enumerate(alleles)], bnp.DNAEncoding)

            row_lengths[1:-1:2] = variant_sequences.shape[1]

            # empty placeholder to fill
            path = bnp.EncodedRaggedArray(bnp.EncodedArray(np.zeros(int(np.sum(row_lengths)), dtype=np.uint8), bnp.DNAEncoding), row_lengths)
            path[0::2] = ref_between
            path[1:-1:2] = variant_sequences
            paths.append(path)

        return Paths(paths, combinations)

    def _make_combination_matrix(self, alleles, n_paths, n_variants):
        # make all possible combinations of variant alleles through the path
        # where all combinations of alleles are present within a window of size self._window
        combinations = itertools.product(*[alleles for _ in range(self._window)])
        # expand combinations to whole genome
        combinations = (itertools.chain(
            *itertools.repeat(c, n_variants // self._window + 1))
            for c in combinations)
        combinations = (itertools.islice(c, n_variants) for c in combinations)
        combination_matrix = np.zeros((n_paths, n_variants), dtype=np.int8)
        for i, c in enumerate(combinations):
            combination_matrix[i] = np.array(list(c))

        return combination_matrix


