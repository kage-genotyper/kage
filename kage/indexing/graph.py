import logging
from dataclasses import dataclass
import bionumpy as bnp
import numpy as np
from kage.preprocessing.variants import VariantAlleleSequences, MultiAllelicVariantSequences
from kage.util import stream_ragged_array
from ..preprocessing.variants import get_padded_variants_from_vcf, Variants
from ..util import zip_sequences
from bionumpy.datatypes import Interval
from typing import Tuple, Union


@dataclass
class GenomeBetweenVariants:
    """ Represents the linear reference genome between variants, not including variant alleles"""
    sequence: bnp.EncodedRaggedArray

    def split(self, k):
        # splits into two GenomeBetweenVariants. The first contains all bases ..
        pass

    def to_list(self):
        return self.sequence.tolist()

    def pad_at_end(self, n_bases):
        s = self.sequence
        shape = s.raw().shape
        shape[1][-1] += n_bases
        self.sequence = bnp.EncodedRaggedArray(np.concatenate([s.ravel(), bnp.as_encoded_array("A" * n_bases, bnp.DNAEncoding)]), shape)

    @classmethod
    def from_list(cls, l):
        return cls(bnp.as_encoded_array(l, bnp.DNAEncoding))

    @classmethod
    def from_reference_and_variant_intervals(cls, reference_sequences: bnp.datatypes.SequenceEntry,
                                             variant_intervals: bnp.datatypes.Interval):
        chromosome_names = reference_sequences.name
        chromosome_sequences = reference_sequences.sequence
        chromosome_lengths = {name.to_string(): len(seq) for name, seq in zip(chromosome_names, chromosome_sequences)}
        global_reference_sequence = np.concatenate(chromosome_sequences)
        global_reference_sequence = bnp.change_encoding(global_reference_sequence, bnp.encodings.ACGTnEncoding)
        global_offset = bnp.genomic_data.global_offset.GlobalOffset(chromosome_lengths)

        variants_global_offset = global_offset.from_local_interval(variant_intervals)
        # start position should be first base of "unique" ref sequence in variant
        # stop should be first base of ref sequence after variant
        global_starts = variants_global_offset.start.copy()
        global_stops = variants_global_offset.stop.copy()
        between_variants_start = np.insert(global_stops, 0, 0)
        between_variants_end = np.insert(global_starts, len(global_starts), len(global_reference_sequence))
        if not np.all(between_variants_start[1:] >= between_variants_start[:-1]):
            logging.error("Some variants start before others end. Are there overlapping variants?")
            raise Exception("")

        # some variants will start and end at the same position. Then we don't need the sequence between them
        adjust_ends = np.where(between_variants_end < between_variants_start)[0]
        logging.info("Adjusting ends for %d variants because these are before starts", len(adjust_ends))
        # these ends should have starts that match the next start
        assert np.all(between_variants_start[adjust_ends] == between_variants_start[adjust_ends + 1])
        # adjust these to be the same as start
        between_variants_end[adjust_ends] = between_variants_start[adjust_ends]
        # between_variants_end = np.maximum(between_variants_start, between_variants_end)

        sequence_between_variants = bnp.ragged_slice(global_reference_sequence, between_variants_start,
                                                     between_variants_end)

        # replace N's with A
        sequence_between_variants[sequence_between_variants == "N"] = "A"
        sequence_between_variants = bnp.change_encoding(sequence_between_variants, bnp.DNAEncoding)

        return cls(sequence_between_variants)


@dataclass
class Graph:
    genome: GenomeBetweenVariants
    variants: Union[VariantAlleleSequences, MultiAllelicVariantSequences]

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

        assert np.all(ref_sequence.shape[1] >= 0), ref_sequence.shape[1]
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
        assert np.all(sequence_lengths >= 0), sequence_lengths
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

        # reading all variants into memory, should be fine with normal vcfs
        logging.info("Reading variants")

        if pad_variants:
            variants = get_padded_variants_from_vcf(vcf_file_name, reference_file_name)
        else:
            variants = bnp.open(vcf_file_name).read()
            variants = Variants.from_vcf_entry(variants)

        return cls.from_variants_and_reference(reference_sequences, variants)


    @classmethod
    def from_variants_and_reference(cls, reference_sequences: bnp.datatypes.SequenceEntry, variants: Variants):

        variant_intervals = Interval(variants.chromosome, variants.position,
                                         variants.position + variants.ref_seq.shape[1])

        genome_between_variants = GenomeBetweenVariants.from_reference_and_variant_intervals(reference_sequences, variant_intervals)

        variant_ref_sequences = variants.ref_seq
        variant_alt_sequences = variants.alt_seq
        variant_allele_sequence = VariantAlleleSequences(np.concatenate([variant_ref_sequences, variant_alt_sequences]))

        return cls(genome_between_variants, variant_allele_sequence)


def make_multiallelic_graph(reference_sequences: bnp.datatypes.SequenceEntry, biallelic_padded_variants: Variants) -> Tuple[Graph, 'VariantAlleleToNodeMap']:
    variant_allele_sequences, node_mapping, variant_regions = biallelic_padded_variants.get_multi_allele_variant_alleles()

    # make intervals for where bubbles start and end
    genome_between_variants = GenomeBetweenVariants.from_reference_and_variant_intervals(reference_sequences, variant_regions)
    return Graph(genome_between_variants, variant_allele_sequences), node_mapping

