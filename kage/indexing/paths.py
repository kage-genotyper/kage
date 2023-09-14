import itertools
import logging
import os
from dataclasses import dataclass
from typing import List, Literal, Union
import bionumpy as bnp
import numpy as np
import tqdm
from shared_memory_wrapper import to_file, from_file
import npstructures as nps


@dataclass
class PathSequences:
    sequences: List[Union['PathSequence', 'DiscBackedPathSequence']]

    def __post_init__(self):
        if isinstance(self.sequences[0], bnp.EncodedRaggedArray):
            print("Converting")
            # Wrap each encoded ragged array in PathSequence
            self.sequences = [PathSequence(s) for s in self.sequences]

    def __getitem__(self, item):
        return self.sequences[item]

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def n_variants(self):
        return self.get_path_sequence(0).shape[0] // 2

    @classmethod
    def from_list(cls, sequences):
        return cls([bnp.as_encoded_array(s) for s in sequences])

    def subset_on_variants(self, from_variant, to_variant, padding=0):
        """
        Subsets sequences on variants by including sequence before the first variant and after the last (noninclusive)
        If padding > 0, will ensure that sequence before first and after last variant is at least padding long
        """
        if isinstance(self.sequences[0], DiscBackedPathSequence):
            # load each and write to discbacked again
            return PathSequences([
                DiscBackedPathSequence.from_non_disc_backed(
                    s.load().subset_on_variants(from_variant, to_variant, padding),
                    f"{s.file}-{from_variant}-{to_variant}")
                for s in self.sequences
            ])
        else:
            return PathSequences(
                [s.subset_on_variants(from_variant, to_variant, padding) for s in self.sequences]
            )
            #return PathSequences([s.sub[from_variant*2:to_variant*2+1] for s in self.iter_path_sequences()])

    def get_path_sequence(self, path_index):
        path_sequence = self.sequences[path_index]
        if isinstance(path_sequence, DiscBackedPathSequence):
            return path_sequence.load()
        else:
            return path_sequence

    def iter_path_sequences(self):
        for i in range(len(self.sequences)):
            yield self.get_path_sequence(i)

    def to_disc_backed(self, file_base_name):
        return PathSequences([
            DiscBackedPathSequence.from_non_disc_backed(path, f"{file_base_name}_path_{i}") for i, path in enumerate(self.sequences)
        ])


@dataclass
class PathCombinationMatrix:
    """
    Represents which allele each path has on each variant (as a matrix)
    """
    matrix: np.ndarray  # n_paths x n_variants

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.uint8)

    def __getitem__(self, item):
        return self.matrix[item]

    @property
    def T(self):
        return self.matrix.T

    @property
    def shape(self):
        return self.matrix.shape

    def __len__(self):
        return len(self.matrix)


@dataclass
class Paths:
    # the sequence for each path, as a ragged array (nodes)
    paths: PathSequences
    # the allele present at each variant in each path
    variant_alleles: PathCombinationMatrix  # n_paths x n_variants

    def n_variants(self):
        return self.variant_alleles.shape[1]

    def subset_on_variants(self, from_id, to_id, padding):
        return Paths(self.paths.subset_on_variants(from_id, to_id, padding),
                     PathCombinationMatrix(self.variant_alleles[:, from_id:to_id]))

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
            assert np.all(window_ends >= window_starts)
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
        self.paths = self.paths.to_disc_backed(file_base_name)

    def remove_tmp_files(self):
        logging.info("Removing tmp files")
        for path in self.paths.sequences:
            if isinstance(path, DiscBackedPathSequence):
                path.remove_file()


@dataclass
class PathSequence:
    """Represents a single path sequence over variants. Basically a wrapper around an EncodedRaggedArray.
    First and last element are always ref sequence before and after last variant.
    Every other element is a variant/ref sequence.
    """
    sequence: bnp.EncodedRaggedArray

    def __getitem__(self, item):
        return self.sequence[item]

    @property
    def _shape(self):
        return self.sequence._shape

    def ravel(self):
        return self.sequence.ravel()

    @property
    def shape(self):
        return self.sequence.shape

    @property
    def size(self):
        return self.sequence.size

    def subset_on_variants(self, from_variant, to_variant, min_padding=0):
        """Subsets on variant and ensures min padding before and after"""
        variant_start_index = 2*from_variant + 1
        variant_end_index = 2*to_variant
        subset = self.sequence[variant_start_index:variant_end_index]

        # Pad with sequence before first
        flat_sequence = self.sequence.ravel()
        start = self.sequence._shape.starts[variant_start_index]
        padding_before = flat_sequence[max(0, start - min_padding):start]
        padding_before = bnp.EncodedRaggedArray(padding_before, [len(padding_before)])

        # padding after
        end = self.sequence._shape.starts[variant_end_index]
        padding_after = flat_sequence[end:min(end + min_padding, len(flat_sequence))]
        padding_after = bnp.EncodedRaggedArray(padding_after, [len(padding_after)])

        return PathSequence(np.concatenate([padding_before, subset, padding_after]))



@dataclass
class DiscBackedPathSequence:
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

    def remove_file(self):
        if os.path.isfile(self.file + ".npz"):
            os.remove(self.file + ".npz")
        else:
            logging.warning("Did not find file %s" % self.file)


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

    def run(self, n_alleles_at_each_variant=None):
        if n_alleles_at_each_variant is None:
            logging.info("Assuming all variants are biallelic")
            alleles = [0, 1]  # possible alleles, assuming biallelic variants
            n_paths = len(alleles)**self._window
            n_variants = self._variants.n_variants
            combinations = PathCreator.make_combination_matrix(alleles, n_variants, self._window)
        else:
            combinations = PathCreator.make_combination_matrix_multi_allele(n_alleles_at_each_variant, self._window)
            n_paths = len(combinations)

        paths = []

        # each path will have sequences between each variant and the allele sequences specified in combinations
        ref_between = self._genome.sequence
        for i, alleles in tqdm.tqdm(enumerate(combinations), total=n_paths, desc="Creating paths through graph", unit="path"):
            # make a new EncodedRaggedArray where every other row is ref/variant
            path = self._graph.sequence(alleles)
            if self._make_disc_backed:
                path = DiscBackedPathSequence.from_non_disc_backed(PathSequence(path), f"{self._disc_backed_file_base_name}_path_{i}")
            paths.append(path)

        return Paths(PathSequences(paths), combinations)

    @staticmethod
    def make_combination_matrix(alleles, n_variants, window) -> PathCombinationMatrix:
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

        return PathCombinationMatrix(combination_matrix)

    @staticmethod
    def make_combination_matrix_multi_allele(n_alleles: np.ndarray, window) -> PathCombinationMatrix:
        """
        Makes a combination matrix for variants with variable number of alleles.
        Only ensures all combinations within a window when alleles are biallelic.
        """
        new = []
        n_alleles = np.asarray(n_alleles)
        total_alleles = np.sum(n_alleles)

        # make biallelic paths first
        biallelic = PathCreator.make_combination_matrix([0, 1], total_alleles, window)
        for i in range(len(biallelic.matrix)):
            path = biallelic.matrix[i]

            grouped_by_variant = PathCreator.convert_biallelic_path_to_multiallelic(n_alleles, path)
            new.append(grouped_by_variant)

        return PathCombinationMatrix(np.array(new))

    @staticmethod
    def convert_biallelic_path_to_multiallelic(n_alleles: np.ndarray, path: np.ndarray, how: Literal["path", "encoding"] = "path"):
        """
        :param n_alleles: number of alleles at each variant
        :param path: allele number at each variant for actual path (or individual)
        :param return_encoded: if True, returns the actual encoded path. If not, takes modulo to get a path (used with PathCreator)
        """
        n_alleles = np.asarray(n_alleles)
        path = np.asarray(path)

        # each multiallelic variant will become one row
        grouped_by_variant = nps.RaggedArray(path, n_alleles - 1)
        multiallele = np.where(grouped_by_variant.shape[1] > 1)[0]
        selection = grouped_by_variant[multiallele]
        # recode each row to one single number between 0 and n_alleles
        # multiple by a factor and do modulo n alles to distribute evenly
        if how == "encoding":
            # encoding is the last variant with positive allele
            factor = nps.RaggedArray([[i+1 for i in range(len(row))] for row in selection])
            grouped_by_variant[multiallele, 0] = np.max(selection * factor, axis=1)  # np.max to get column index
        else:
            # encoding is using modulo to just distribute the path over all alleles evenly
            factor = nps.RaggedArray([[2 ** i for i in range(len(row))] for row in selection])
            encoded_multiallelic_path = np.sum(selection * factor, axis=1)
            selection = encoded_multiallelic_path % (selection.shape[1] + 1)
            # we want all nonmultiallelic and the new encoding for the multiallelic
            # hack: Set first column of ragged array to new and slice
            grouped_by_variant[multiallele, 0] = selection
        return grouped_by_variant[:, 0]


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
