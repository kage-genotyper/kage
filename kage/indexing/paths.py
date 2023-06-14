import itertools
import logging
from dataclasses import dataclass
from typing import List

import bionumpy as bnp
import numpy as np
import tqdm
from shared_memory_wrapper import to_file, from_file


@dataclass
class Paths:
    # the sequence for each path, as a ragged array (nodes)
    paths: List[bnp.EncodedRaggedArray]
    # the allele present at each variant in each path
    variant_alleles: np.ndarray  # n_paths x n_variants

    def get_path_sequence(self, path_index):
        path_sequence = self.paths[path_index]
        if isinstance(path_sequence, DiscBackedPath):
            return path_sequence.load()
        else:
            return path_sequence

    def iter_path_sequences(self):
        for i in range(len(self.paths)):
            yield self.get_path_sequence(i)

    def n_variants(self):
        return self.variant_alleles.shape[1]

    def paths_for_allele_at_variant(self, allele, variant):
        relevant_paths = np.where(self.variant_alleles[:, variant] == allele)[0]
        return [self.paths[p] for p in relevant_paths]

    def get_windows_around_variants(self, k):
        logging.info("Computing kmers around variants")
        windowed_paths = []
        for i, path in enumerate(self.paths):
            starts = path._shape.starts[1::2]
            window_starts = starts - k + 1
            window_ends = np.minimum(starts + k, path.size)
            windows = bnp.ragged_slice(path.ravel(), window_starts, window_ends)
            windowed_paths.append(
                bnp.get_kmers(windows, k=k)
            )
        return VariantWindowKmers(windowed_paths, self.variant_alleles)

    def __str__(self):
        return "\n".join(f"Path {i}: {path}" for i, path in enumerate(self.paths))

    def __repr__(self):
        return str(self)

    def get_kmers(self, variant, allele, kmer_size=3):
        raise NotImplementedError("Not to be used more")
        # gets all kmers of size kmer_size around the variant allele (on all paths relevant for the allele)
        path_sequences = self.paths_for_allele_at_variant(allele, variant)
        n_kmers_per_window = kmer_size
        out = np.zeros((len(path_sequences), n_kmers_per_window), dtype=np.uint64)
        for i, path in enumerate(path_sequences):
            offset_in_path = path._shape.starts[variant * 2 + 1]
            start = offset_in_path - kmer_size + 1
            assert start >= 0, "NOt supporting variants < kmer_size away from edge"
            end = offset_in_path + kmer_size
            kmers = bnp.get_kmers(path.ravel()[start:end], kmer_size).ravel().raw().astype(np.uint64)
            out[i, :len(kmers)] = kmers

        return out

    def to_disc_backend(self, file_base_name):
        self.paths = [
            DiscBackedPath(to_file(path, f"{file_base_name}_path_{i}")) for i, path in enumerate(self.paths)
        ]


@dataclass
class DiscBackedPath:
    """
    Can be used as the paths property in Paths, stores sequences on disk and only allows iterating over them.
    """
    # the sequence for each path, as a ragged array (nodes)
    file: str

    def __getitem__(self, item):
        data = from_file(self.file)
        return data[item]

    def load(self):
        return from_file(self.file)

    @classmethod
    def from_non_disc_backed(cls, path, file_name):
        return cls(to_file(path, file_name))


class PathCreator:
    def __init__(self, graph, window: int = 3, make_disc_backed=False, disc_backed_file_base_name=None):
        self._graph = graph
        self._variants = graph.variants
        self._genome = graph.genome
        self._window = window
        self._make_disc_backed = make_disc_backed
        if self._make_disc_backed:
            logging.info("Will make disc backed paths")
            assert disc_backed_file_base_name is not None
            self._disc_backed_file_base_name = disc_backed_file_base_name

    def run(self):
        alleles = [0, 1]  # possible alleles, assuming biallelic variants
        n_paths = len(alleles)**self._window
        n_variants = self._variants.n_variants
        combinations = PathCreator.make_combination_matrix(alleles, n_variants, self._window)

        paths = []

        # each path will have sequences between each variant and the allele sequences specified in combinations
        ref_between = self._genome.sequence
        for i, alleles in tqdm.tqdm(enumerate(combinations), total=n_paths, desc="Creating paths through graph", unit="path"):
            # make a new EncodedRaggedArray where every other row is ref/variant
            path = self._graph.sequence(alleles)
            if self._make_disc_backed:
                path = DiscBackedPath.from_non_disc_backed(path, f"{self._disc_backed_file_base_name}_path_{i}")
            paths.append(path)

        return Paths(paths, combinations)

    @staticmethod
    def make_combination_matrix(alleles, n_variants, window):
        n_paths = len(alleles)**window
        # make all possible combinations of variant alleles through the path
        # where all combinations of alleles are present within a window of size self._window
        combinations = itertools.product(*[alleles for _ in range(window)])
        # expand combinations to whole genome
        combinations = (itertools.chain(
            *itertools.repeat(c, n_variants // window + 1))
            for c in combinations)
        combinations = (itertools.islice(c, n_variants) for c in combinations)
        combination_matrix = np.zeros((n_paths, n_variants), dtype=np.int8)
        for i, c in enumerate(combinations):
            combination_matrix[i] = np.array(list(c))

        return combination_matrix


@dataclass
class VariantWindowKmers:
    """
    Kmers for variants.
    """
    kmers: List[bnp.EncodedRaggedArray]  # each element is a path, rows in ragged array are variants
    variant_alleles: np.ndarray  # the allele present at each variant in each path

    def get_kmers(self, variant, allele):
        relevant_paths = np.where(self.variant_alleles[:, variant] == allele)[0]
        n_kmers_per_window = len(self.kmers[relevant_paths[0]][variant])
        out = np.zeros((len(relevant_paths), n_kmers_per_window), dtype=np.uint64)
        for i, path in enumerate(relevant_paths):
            out[i, :] = self.kmers[path][variant].raw()
        return out
