import itertools
import logging
import time
from dataclasses import dataclass
from typing import List
import npstructures as nps
import numpy as np
import bionumpy as bnp
from graph_kmer_index import KmerIndex, FlatKmers
import tqdm
from shared_memory_wrapper import from_file, to_file

from .sparse_haplotype_matrix import SparseHaplotypeMatrix
from ..models.mapping_model import LimitedFrequencySamplingComboModel
from .tricky_variants import TrickyVariants
from bionumpy.datatypes import Interval

from ..preprocessing.variants import Variants, get_padded_variants_from_vcf
from ..util import log_memory_usage_now
from shared_memory_wrapper.util import interval_chunks

"""
Module for simple variant signature finding by using static predetermined paths through the "graph".

Works when all variants are biallelic and there are no overlapping variants.
"""


@dataclass
class GenomeBetweenVariants:
    """ Represents the linear reference genome between variants, not including variant alleles"""
    sequence: bnp.EncodedRaggedArray

    def split(self, k):
        # splits into two GenomeBetweenVariants. The first contains all bases ..
        pass


class VariantAlleleSequences:
    def __init__(self, data: bnp.EncodedRaggedArray, n_alleles: int = 2):
        self._data = data  # data contains sequence for first allele first, then second allele, etc.
        self.n_alleles = n_alleles
        self.n_variants = len(self._data) // n_alleles

    @classmethod
    def from_list(cls, variant_sequences: List[List]):
        zipped = list(itertools.chain(*zip(*variant_sequences)))
        encoded = bnp.as_encoded_array(zipped, bnp.encodings.ACGTnEncoding)
        encoded[encoded == "N"] = "A"
        return cls(bnp.change_encoding(encoded, bnp.DNAEncoding))

    @property
    def allele_sequences(self):
        return [self._data[self.n_variants*allele:self.n_variants*(allele+1)] for allele in range(self.n_alleles)]

    def get_allele_sequence(self, variant, allele):
        return self._data[self.n_variants*allele + variant].to_string()

    def get_haplotype_sequence(self, haplotypes: np.ndarray) -> bnp.EncodedRaggedArray:
        """
        Gets all sequences from nodes given the haplotypes at those nodes.
        """
        assert len(haplotypes) == self.n_variants
        rows = self.n_variants*haplotypes + np.arange(self.n_variants)
        return self._data[rows]


def zip_sequences(a: bnp.EncodedRaggedArray, b: bnp.EncodedRaggedArray):
    """Utility function for merging encoded ragged arrays ("zipping" rows)"""
    assert len(a) == len(b)+1

    row_lengths = np.zeros(len(a)+len(b))
    row_lengths[0::2] = a.shape[1]
    row_lengths[1::2] = b.shape[1]

    # empty placeholder to fill

    new = bnp.EncodedRaggedArray(
        bnp.EncodedArray(np.zeros(int(np.sum(row_lengths)), dtype=np.uint8), bnp.DNAEncoding),
        row_lengths)
    new[0::2] = a
    new[1:-1:2] = b
    return new


def stream_ragged_array(a, chunk_size=100000):
    chunks = interval_chunks(0, len(a), 1+len(a) // chunk_size)
    for chunk in chunks:
        yield a[chunk[0]:chunk[1]]

@dataclass
class Graph:
    genome: GenomeBetweenVariants
    variants: VariantAlleleSequences

    def n_variants(self):
        return self.variants.n_variants

    def n_nodes(self):
        return self.n_variants()*2

    def sequence(self, haplotypes: np.ndarray, stream=False) -> bnp.EncodedArray:
        """
        Returns the sequence through the graph given haplotypes for all variants
        """
        assert len(haplotypes) == self.n_variants(), (len(haplotypes), self.n_variants())
        ref_sequence = self.genome.sequence
        variant_sequences = self.variants.get_haplotype_sequence(haplotypes)

        assert np.all(ref_sequence.shape[1] >= 0)
        assert np.all(variant_sequences.shape[1] >= 0)

        # stitch these together
        result = zip_sequences(ref_sequence, variant_sequences)
        assert np.all(result.shape[1] >= 0), result.shape[1]
        if not stream:
            return result
        else:
            return (res for res in stream_ragged_array(result))

    def sequence_of_pairs_of_ref_and_variants_as_ragged_array(self, haplotypes: np.ndarray) -> bnp.EncodedRaggedArray:
        """
        Every row in the returned RaggedArray is the sequence for a variant and the next reference segment.
        The first row is the first reference segment.
        """
        sequence = self.sequence(haplotypes)
        # merge pairs of rows
        old_lengths = sequence.shape[1]
        assert np.all(old_lengths >= 0)
        new_lengths = np.zeros(1+len(sequence)//2, dtype=int)
        new_lengths[0] = old_lengths[0]
        new_lengths[1:] = old_lengths[1::2] + old_lengths[2::2]
        return bnp.EncodedRaggedArray(sequence.ravel(), new_lengths)

    def kmers_for_pairs_of_ref_and_variants(self, haplotypes: np.ndarray, k: int) -> bnp.EncodedRaggedArray:
        """
        Returns a ragged array where each row is the kmers for a variant allele (given by the haplotypes)
        and the next ref sequence. The first element is only the first sequence in the graph.
        """
        sequences = self.sequence_of_pairs_of_ref_and_variants_as_ragged_array(haplotypes)
        all_kmers = bnp.get_kmers(sequences.ravel(), k)
        sequence_lengths = sequences.shape[1].copy()
        assert sequence_lengths[-1] >= k, "Last sequence in graph must be larger than k"
        # on last node, there will be fewer kmers
        sequence_lengths[-1] -= k-1
        assert np.all(sequence_lengths > 0)
        return bnp.EncodedRaggedArray(all_kmers.ravel(), sequence_lengths)

    def get_haplotype_kmers(self, haplotype: np.array, k, stream=False) -> np.ndarray:
        if not stream:
            sequence = self.sequence(haplotype).ravel()
            return bnp.get_kmers(sequence, k).ravel().raw().astype(np.uint64)
        else:
            sequence = self.sequence(haplotype, stream=True)
            return (bnp.get_kmers(subseq.ravel(), k).ravel().raw().astype(np.uint64) for subseq in sequence)

    @classmethod
    def from_vcf(cls, vcf_file_name, reference_file_name, pad_variants=False):
        reference_sequences = bnp.open(reference_file_name).read()
        chromosome_names = reference_sequences.name
        chromosome_sequences = reference_sequences.sequence
        chromosome_lengths = {name.to_string(): len(seq) for name, seq in zip(chromosome_names, chromosome_sequences)}

        global_reference_sequence = np.concatenate(chromosome_sequences)
        global_reference_sequence = bnp.change_encoding(global_reference_sequence, bnp.encodings.ACGTnEncoding)

        global_offset = bnp.genomic_data.global_offset.GlobalOffset(chromosome_lengths)

        # reading all variants into memory, should be fine with normal vcfs
        logging.info("Reading variants")

        if pad_variants:
            variants = get_padded_variants_from_vcf(vcf_file_name, reference_file_name)
        else:
            variants = bnp.open(vcf_file_name).read()
            variants = Variants.from_vcf_entry(variants)

        variants_as_intervals = Interval(variants.chromosome, variants.position, variants.position+variants.ref_seq.shape[1])
        variants_global_offset = global_offset.from_local_interval(variants_as_intervals)

        # start position should be first base of "unique" ref sequence in variant
        # stop should be first base of ref sequence after variant
        global_starts = variants_global_offset.start.copy()
        global_stops = variants_global_offset.stop.copy()

        between_variants_start = np.insert(global_stops, 0, 0)
        between_variants_end = np.insert(global_starts, len(global_starts), len(global_reference_sequence))

        assert np.all(between_variants_start <= between_variants_end), "Some variants start before others end. Are there overlapping variants?"

        sequence_between_variants = bnp.ragged_slice(global_reference_sequence, between_variants_start, between_variants_end)


        variant_ref_sequences = variants.ref_seq
        variant_alt_sequences = variants.alt_seq

        # replace N's with A
        sequence_between_variants[sequence_between_variants == "N"] = "A"
        sequence_between_variants = bnp.change_encoding(sequence_between_variants, bnp.DNAEncoding)



        return cls(GenomeBetweenVariants(sequence_between_variants),
                   VariantAlleleSequences(np.concatenate([variant_ref_sequences, variant_alt_sequences])))


    @classmethod
    def _from_vcf(cls, vcf_file_name, reference_file_name, k=31):
        reference = bnp.open(reference_file_name, bnp.encodings.ACGTnEncoding).read()
        reference = {s.name.to_string(): s.sequence for s in reference}
        #assert len(reference) == 1, "Only one chromosome supported now"


        sequences_between_variants = []
        variant_sequences = []

        vcf = bnp.open(vcf_file_name, bnp.DNAEncoding)
        prev_ref_pos = 0
        prev_chromosome = None
        for chunk in vcf:
            for variant in chunk:
                chromosome = variant.chromosome.to_string()
                pos = variant.position
                ref = variant.ref_seq
                alt = variant.alt_seq

                #if pos > len(reference) - k:
                #logging.info("Skipping variant too close to end of chromosome")
                #continue


                if len(ref) > len(alt) or len(alt) > len(ref):
                    # indel
                    pos_before = pos
                    if len(ref) == 1:
                        # insertion
                        pos_after = pos_before + 1
                        ref = ref[0:0]
                        alt = alt[1:]
                    else:
                        # deletion
                        assert len(alt) == 1
                        pos_after = pos_before + len(ref)
                        alt = alt[0:0]
                        ref = ref[1:]

                else:
                    # snp
                    assert len(ref) == len(alt) == 1
                    pos_before = pos - 1
                    pos_after = pos_before + 2

                if prev_chromosome is not None and chromosome != prev_chromosome:
                    # add last bit of last chromosome
                    new_sequence = reference[prev_chromosome][prev_ref_pos:].to_string().upper().replace("N", "A") \
                        + reference[chromosome][0:pos_before + 1].to_string().upper().replace("N", "A")
                    sequences_between_variants.append(new_sequence)
                else:
                    new_sequence = reference[chromosome][prev_ref_pos:pos_before + 1].to_string().upper().replace("N",
                                                                                                                  "A")
                    sequences_between_variants.append(new_sequence)

                prev_ref_pos = pos_after
                variant_sequences.append([ref.to_string(), alt.to_string()])

                prev_chromosome = chromosome

        # add last bit of reference
        sequences_between_variants.append(reference[chromosome][prev_ref_pos:].to_string().upper().replace("N", "A"))

        return cls(GenomeBetweenVariants(bnp.as_encoded_array(sequences_between_variants, bnp.DNAEncoding)),
                   VariantAlleleSequences.from_list(variant_sequences))


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


@dataclass
class MatrixVariantWindowKmers:
    """
    Represents kmers around variants in a 3-dimensional matrix, possible when n kmers per variant allele is fixed
    """
    kmers: np.ndarray  # n_paths, n variants x n_windows
    variant_alleles: np.ndarray

    @classmethod
    def from_paths(cls, paths, k):
        n_variants = paths.n_variants()
        print(f"n_variants: {n_variants}")
        n_paths = len(paths.paths)
        n_windows = k - k//2 - 1
        print(f"n_paths: {n_paths}, n_windows: {n_windows}")
        matrix = np.zeros((n_paths, n_variants, n_windows), dtype=np.uint64)


        for i, path in enumerate(paths.iter_path_sequences()):
            starts = path._shape.starts[1::2]
            window_starts = starts - k + 1 + k//2
            #window_ends = np.minimum(starts + k, path.size)
            window_ends = starts + k - 1
            windows = bnp.ragged_slice(path.ravel(), window_starts, window_ends)
            kmers = bnp.get_kmers(windows, k=k).ravel().raw().astype(np.uint64)
            kmers = kmers.reshape((n_variants, n_windows))
            matrix[i, :, :] = kmers

        return cls(matrix, paths.variant_alleles)

    def get_kmers(self, variant, allele):
        relevant_paths = np.where(self.variant_alleles[:, variant] == allele)[0]
        return self.kmers[relevant_paths, variant, :]

    def get_best_kmers(self, scorer):
        # score all kmers
        # returns a matrix of shape n_paths x n_variants containing the kmers chosen for each variant
        # pick one window per variant
        logging.info("Scoring all candidate kmers")
        scores = scorer.score_kmers(self.kmers.ravel()).reshape(self.kmers.shape)
        # window scores are the lowest score among all paths in windows
        #scores = np.min(scores, axis=0)
        scores = np.sum(scores, axis=0)
        # score rightmost window a bit more to prefer this on a tie
        scores[:, -1] += 1
        # best window has the highest score
        logging.info("Finding best windows based on scores")
        best_windows = np.argmax(scores, axis=1)
        best_kmers = self.kmers[:, np.arange(self.kmers.shape[1]), best_windows]
        assert best_kmers.shape[0] == self.kmers.shape[0]
        return best_kmers


class FastApproxCounter:
    """ Fast counter that uses modulo and allows collisions"""
    def __init__(self, array, modulo):
        self._array = array
        self._modulo = modulo

    @classmethod
    def empty(cls, modulo):
        return cls(np.zeros(modulo, dtype=np.uint16), modulo)

    def add(self, values):
        self._array[values % self._modulo] += 1

    @classmethod
    def from_keys_and_values(cls, keys, values, modulo):
        array = np.zeros(modulo, dtype=np.uint16)
        array[keys % modulo] = values
        return cls(array, modulo)

    def __getitem__(self, keys):
        return self._array[keys % self._modulo]

    def score_kmers(self, kmers):
        return -self[kmers]


def make_kmer_scorer_from_random_haplotypes(graph: Graph, haplotype_matrix: SparseHaplotypeMatrix,
                                            k: int,
                                            n_haplotypes: int = 4,
                                            modulo: int = 20000033):
    """
    Estimates counts from random individuals
    """
    counter = FastApproxCounter.empty(modulo)
    chosen_haplotypes = np.random.choice(np.arange(haplotype_matrix.n_haplotypes), n_haplotypes, replace=False)
    logging.info("Picked random haplotypes to make kmer scorer: %s" % chosen_haplotypes)
    haplotype_nodes = (haplotype_matrix.get_haplotype(haplotype) for haplotype in chosen_haplotypes)

    # also add the reference and a haplotype with all variants
    haplotype_nodes = itertools.chain(haplotype_nodes,
                                      [np.zeros(haplotype_matrix.n_variants, dtype=np.uint8),
                                        np.ones(haplotype_matrix.n_variants, dtype=np.uint8)])

    for i, nodes in tqdm.tqdm(enumerate(haplotype_nodes), desc="Estimating global kmer counts", total=len(chosen_haplotypes), unit="haplotype"):
        #haplotype_nodes = haplotype_matrix.get_haplotype(haplotype)
        log_memory_usage_now("Memory after getting nodes")
        kmers = graph.get_haplotype_kmers(nodes, k=k, stream=True)
        log_memory_usage_now("Memory after kmers")
        for subkmers in kmers:
            counter.add(subkmers)
        log_memory_usage_now("After adding haplotype %d" % i)

    # also add the reference and a haplotype with all variants
    #kmers = graph.get_haplotype_kmers(np.zeros(haplotype_matrix.n_variants, dtype=np.uint8), k=k, stream=False)
    #print(kmers)
    #counter.add(kmers)
    #counter.add(graph.get_haplotype_kmers(np.ones(haplotype_matrix.n_variants, dtype=np.uint8), k=k))

    return counter

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


class Scorer:
    def __init__(self):
        pass

    def score_kmers(self, kmers):
        pass


class KmerFrequencyScorer(Scorer):
    def __init__(self, frequency_lookup: nps.HashTable):
        self._frequency_lookup = frequency_lookup

    def score_kmers(self, kmers):
        return np.max(self._frequency_lookup[kmers])



@dataclass
class Signatures:
    ref: nps.RaggedArray
    alt: nps.RaggedArray

    def get_overlap(self):
        return np.sum(np.in1d(self.ref.ravel(), self.alt.ravel())) / len(self.ref.ravel())

    def get_as_flat_kmers(self):
        signatures = self
        logging.info("Signature overlap: %.5f percent" % (signatures.get_overlap() * 100))

        all_kmers = []
        all_node_ids = []

        for variant_id, (ref_kmers, variant_kmers) in enumerate(zip(signatures.ref, signatures.alt)):
            node_id_ref = variant_id * 2
            node_id_alt = variant_id * 2 + 1
            all_kmers.append(ref_kmers)
            all_node_ids.append(np.full(len(ref_kmers), node_id_ref, dtype=np.uint32))
            all_kmers.append(variant_kmers)
            all_node_ids.append(np.full(len(variant_kmers), node_id_alt, dtype=np.uint32))

        from graph_kmer_index import FlatKmers
        return FlatKmers(np.concatenate(all_kmers), np.concatenate(all_node_ids))

    def get_as_kmer_index(self, k, modulo=103, include_reverse_complements=True):
        flat = self.get_as_flat_kmers()
        if include_reverse_complements:
            rev_comp_flat = flat.get_reverse_complement_flat_kmers(k)
            flat = FlatKmers.from_multiple_flat_kmers([flat, rev_comp_flat])

        assert modulo > len(flat._hashes), "Modulo is too small for number of kmers. Increase"
        index = KmerIndex.from_flat_kmers(flat, skip_frequencies=True, modulo=modulo)
        index.convert_to_int32()
        return index


class SignatureFinder:
    """
    For each variant chooses a signature (a set of kmers) based one a Scores
    """
    def __init__(self, paths: Paths, scorer: Scorer = None, k=3):
        self._paths = paths
        self._scorer = scorer
        self._k = k

    def run(self) -> Signatures:
        """
        Returns a list of kmers. List contains alleles, each RaggedArray represents the variants
        """
        chosen_ref_kmers = []
        chosen_alt_kmers = []

        for variant in tqdm.tqdm(range(self._paths.n_variants()), desc="Finding signatures", unit="variants", total=self._paths.n_variants()):
            best_kmers = None
            best_score = -10000000000000000000
            all_ref_kmers = self._paths.get_kmers(variant, 0, self._k)
            all_alt_kmers = self._paths.get_kmers(variant, 1, self._k)

            assert np.all(all_ref_kmers.shape[1] == all_alt_kmers.shape[1])

            for window in range(len(all_ref_kmers[0])):
                ref_kmers = np.unique(all_ref_kmers[:, window]) #.raw().ravel())
                alt_kmers = np.unique(all_alt_kmers[:, window]) #.raw().ravel())

                all_kmers = np.concatenate([ref_kmers, alt_kmers])
                window_score = self._scorer.score_kmers(all_kmers) if self._scorer is not None else 0

                # if overlap between ref and alt, lower score
                if len(set(ref_kmers).intersection(alt_kmers)) > 0:
                    window_score -= 1000

                if window_score > best_score:
                    best_kmers = [ref_kmers, alt_kmers]
                    best_score = window_score


            chosen_ref_kmers.append(best_kmers[0])
            chosen_alt_kmers.append(best_kmers[1])

        return Signatures(nps.RaggedArray(chosen_ref_kmers), nps.RaggedArray(chosen_alt_kmers))


class SignatureFinder2(SignatureFinder):
    def run(self) -> Signatures:
        # for each path, first creates a RaggedArray with windows for each variant
        paths = MatrixVariantWindowKmers.from_paths(self._paths, self._k)  # ._paths.get_windows_around_variants(self._k)
        chosen_ref_kmers = []
        chosen_alt_kmers = []

        for variant in tqdm.tqdm(range(self._paths.n_variants()), desc="Finding signatures", unit="variants",
                                 total=self._paths.n_variants()):
            best_kmers = None
            best_score = -10000000000000000000
            all_ref_kmers = paths.get_kmers(variant, 0)
            all_alt_kmers = paths.get_kmers(variant, 1)
            #assert np.all(all_ref_kmers.shape[1] == all_alt_kmers.shape[1])

            for window in range(len(all_ref_kmers[0])):
                ref_kmers = np.unique(all_ref_kmers[:, window])  # .raw().ravel())
                alt_kmers = np.unique(all_alt_kmers[:, window])  # .raw().ravel())

                all_kmers = np.concatenate([ref_kmers, alt_kmers])
                window_score = self._scorer.score_kmers(all_kmers) if self._scorer is not None else 0

                # if overlap between ref and alt, lower score
                if len(set(ref_kmers).intersection(alt_kmers)) > 0:
                    window_score -= 1000

                if window_score > best_score:
                    best_kmers = [ref_kmers, alt_kmers]
                    best_score = window_score

            chosen_ref_kmers.append(best_kmers[0])
            chosen_alt_kmers.append(best_kmers[1])

        return Signatures(nps.RaggedArray(chosen_ref_kmers), nps.RaggedArray(chosen_alt_kmers))


class SignatureFinder3(SignatureFinder):
    def run(self) -> Signatures:
        kmers = MatrixVariantWindowKmers.from_paths(self._paths, self._k)
        best_kmers = kmers.get_best_kmers(self._scorer)

        # hack to split matrix into ref and alt kmers
        n_variants = self._paths.n_variants()
        n_paths = len(self._paths.paths)
        all_ref_kmers = best_kmers.T[kmers.variant_alleles.T == 0].reshape((n_variants, n_paths//2))
        all_alt_kmers = best_kmers.T[kmers.variant_alleles.T == 1].reshape((n_variants, n_paths//2))

        chosen_ref_kmers = []
        chosen_alt_kmers = []

        for variant_id, (ref_kmers, alt_kmers) in tqdm.tqdm(enumerate(zip(all_ref_kmers, all_alt_kmers)), desc="Iterating signatures", unit="variants", total=n_variants):
            ref_kmers = np.unique(ref_kmers)
            alt_kmers = np.unique(alt_kmers)
            if set(ref_kmers).intersection(alt_kmers):
                ref_kmers = np.array([])
                alt_kmers = np.array([])

            chosen_ref_kmers.append(ref_kmers)
            chosen_alt_kmers.append(alt_kmers)

        return Signatures(nps.RaggedArray(chosen_ref_kmers), nps.RaggedArray(chosen_alt_kmers))


class SignaturesWithNodes:
    """
    Represents signatures compatible with graph-nodes.
    Nodes are given implicitly from variant ids. Makes it possible
    to create a kmer index that requires nodes.
    """
    pass


class MappingModelCreator:
    def __init__(self, graph: Graph, kmer_index: KmerIndex,
                 haplotype_matrix: SparseHaplotypeMatrix,
                 max_count=10, k=31):
        self._graph = graph
        self._kmer_index = kmer_index
        self._haplotype_matrix = haplotype_matrix
        self._n_nodes = graph.n_nodes()
        self._counts = LimitedFrequencySamplingComboModel.create_empty(self._n_nodes, max_count)
        logging.info("Inited empty model")
        self._k = k
        self._max_count = max_count

    def _process_individual(self, i):
        logging.info("Processing individual %d" % i)
        t0 = time.perf_counter()
        # extract kmers from both haplotypes and map these using the kmer index
        haplotype1 = self._haplotype_matrix.get_haplotype(i*2)
        haplotype2 = self._haplotype_matrix.get_haplotype(i*2+1)
        logging.info("Got haplotypes, took %.2f seconds" % (time.perf_counter() - t0))

        haplotype1_nodes = self._haplotype_matrix.get_haplotype_nodes(i*2)
        haplotype2_nodes = self._haplotype_matrix.get_haplotype_nodes(i*2+1)


        sequence1 = self._graph.sequence(haplotype1).ravel()
        sequence2 = self._graph.sequence(haplotype2).ravel()
        logging.info("Got haplotypes, total now: %.2f seconds" % (time.perf_counter() - t0))

        kmers1 = bnp.get_kmers(sequence1, self._k).ravel().raw().astype(np.uint64)
        kmers2 = bnp.get_kmers(sequence2, self._k).ravel().raw().astype(np.uint64)
        logging.info("Got kmers, total now: %.2f seconds" % (time.perf_counter() - t0))

        t0_mapping_kmers = time.perf_counter()
        node_counts = self._kmer_index.map_kmers(kmers1, self._n_nodes)
        node_counts += self._kmer_index.map_kmers(kmers2, self._n_nodes)
        logging.info("Mapped kmers, took: %.2f seconds" % (time.perf_counter() - t0_mapping_kmers))

        self._add_node_counts(haplotype1_nodes, haplotype2_nodes, node_counts)
        logging.info("Done, total now: %.2f seconds" % (time.perf_counter() - t0))

    def _add_node_counts(self, haplotype1_nodes, haplotype2_nodes, node_counts):
        # split into nodes that the haplotype has and nodes not
        # mask represents the number of haplotypes this individual has per node (0, 1 or 2 for diploid individuals)
        mask = np.zeros(self._n_nodes, dtype=np.int8)
        mask[haplotype1_nodes] += 1
        mask[haplotype2_nodes] += 1
        for genotype in [0, 1, 2]:
            nodes_with_genotype = np.where(mask == genotype)[0]
            counts_on_nodes = node_counts[nodes_with_genotype].astype(int)
            below_max_count = np.where(counts_on_nodes < self._max_count)[
                0]  # ignoring counts larger than supported by matrix
            self._counts.diplotype_counts[genotype][
                nodes_with_genotype[below_max_count], counts_on_nodes[below_max_count]
            ] += 1

    def run(self) -> LimitedFrequencySamplingComboModel:
        n_variants, n_haplotypes = self._haplotype_matrix.shape
        n_nodes = n_variants * 2
        for individual in tqdm.tqdm(range(n_haplotypes // 2), total=n_haplotypes // 2, desc="Creating count model", unit="individuals"):
            self._process_individual(individual)

        return self._counts




def make_linear_reference_kmer_counter(reference_file_name, k=31, modulo=20000033):
    """
    Gets all kmers from a linear reference.
    """
    from graph_kmer_index.kmer_counter import KmerCounter
    logging.info("Getting kmers from linear ref")
    reference = bnp.open(reference_file_name).read()
    reference.sequence[reference.sequence == "N"] = "A"  # tmp hack to allow kmer encoding

    kmers = bnp.get_kmers(reference.sequence, k)
    kmers = kmers.raw().ravel().astype(np.uint64)
    
    from graph_kmer_index.kmer_hashing import kmer_hashes_to_reverse_complement_hash
    logging.info("Getting reverse complements")
    revcomp = kmer_hashes_to_reverse_complement_hash(kmers, k)
    kmers = np.concatenate([kmers, revcomp])

    counter = KmerCounter.from_kmers(kmers, modulo)
    return counter



def make_scorer_from_paths(paths: Paths, k, modulo):
    kmers = []
    for path in tqdm.tqdm(paths.paths, desc="Getting kmers from paths", unit="paths"):
        kmers.append(bnp.get_kmers(path.ravel(), k=k).raw().ravel().astype(np.uint64))
        kmers.append(bnp.get_kmers(bnp.sequence.get_reverse_complement(path).ravel(), k=k).raw().ravel().astype(np.uint64))

    kmers = np.concatenate(kmers).astype(np.int64)

    # using fastapproccounter, we only need to bincount
    counter = np.bincount(kmers % modulo, minlength=modulo)
    return FastApproxCounter(counter, modulo=modulo)

    #logging.info("Getting unique kmers and counts")
    #unique, counts = np.unique(kmers, return_counts=True)
    #return FastApproxCounter.from_keys_and_values(unique, counts, modulo=modulo)


def find_tricky_variants_from_signatures(signatures: Signatures):
    """
    Finds variants with bad signatures.
    """
    nonunique_ref = np.in1d(signatures.ref.ravel(), signatures.alt.ravel())
    nonunique_alt = np.in1d(signatures.alt.ravel(), signatures.ref.ravel())
    #mask_ref = np.zeros_like(signatures.ref, dtype=bool)
    #mask_ref[nonunique_ref] = True
    mask_ref = nps.RaggedArray(nonunique_ref, signatures.ref.shape)
    mask_alt = nps.RaggedArray(nonunique_alt, signatures.alt.shape)
    print(nonunique_ref, nonunique_alt)
    #mask_alt = np.zeros_like(signatures.alt, dtype=bool)
    #mask_alt[nonunique_alt] = True

    tricky_variants_ref = np.any(mask_ref, axis=1)
    tricky_variants_alt = np.any(mask_alt, axis=1)
    tricky_variants = np.logical_or(tricky_variants_ref, tricky_variants_alt)

    logging.info(f"{np.sum(tricky_variants)} tricky variants")
    return TrickyVariants(tricky_variants)


