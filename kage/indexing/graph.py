import itertools
import logging
from dataclasses import dataclass
from typing import List
import bionumpy as bnp
import numpy as np
from kage.util import stream_ragged_array
from ..preprocessing.variants import get_padded_variants_from_vcf, Variants
from ..util import zip_sequences
from bionumpy.datatypes import Interval
import awkward as ak


@dataclass
class GenomeBetweenVariants:
    """ Represents the linear reference genome between variants, not including variant alleles"""
    sequence: bnp.EncodedRaggedArray

    def split(self, k):
        # splits into two GenomeBetweenVariants. The first contains all bases ..
        pass

    def pad_at_end(self, n_bases):
        s = self.sequence
        shape = s.raw().shape
        shape[1][-1] += n_bases
        self.sequence = bnp.EncodedRaggedArray(np.concatenate([s.ravel(), bnp.as_encoded_array("A" * n_bases, bnp.DNAEncoding)]), shape)

    @classmethod
    def from_list(cls, l):
        return cls(bnp.as_encoded_array(l, bnp.DNAEncoding))

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


class MultiAllelicVariantSequences(VariantAlleleSequences):
    """Supports more than two alleles (and flexible number of alleles for each variant).
    Uses awkward arrays internally"""

    def __init__(self, data: ak.Array):
        self._data = data
        self.n_variants = len(self._data)

    @classmethod
    def from_list(cls, variant_sequences: List[List]):
        return cls(ak.Array(variant_sequences))

    def get_haplotype_sequence(self, haplotypes: np.ndarray) -> bnp.EncodedRaggedArray:
        assert len(haplotypes) == self.n_variants
        allele_sequences = self._data[np.arange(self.n_variants), haplotypes]
        shape = ak.to_numpy(ak.num(allele_sequences))
        # flatten, and encode
        bytes = ak.to_numpy(ak.flatten(ak.without_parameters(allele_sequences)))
        return bnp.EncodedRaggedArray(bnp.change_encoding(bnp.EncodedArray(bytes, bnp.BaseEncoding), bnp.DNAEncoding), shape)


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
        chromosome_names = reference_sequences.name
        chromosome_sequences = reference_sequences.sequence
        chromosome_lengths = {name.to_string(): len(seq) for name, seq in zip(chromosome_names, chromosome_sequences)}
        global_reference_sequence = np.concatenate(chromosome_sequences)
        global_reference_sequence = bnp.change_encoding(global_reference_sequence, bnp.encodings.ACGTnEncoding)
        global_offset = bnp.genomic_data.global_offset.GlobalOffset(chromosome_lengths)
        variants_as_intervals = Interval(variants.chromosome, variants.position,
                                         variants.position + variants.ref_seq.shape[1])
        variants_global_offset = global_offset.from_local_interval(variants_as_intervals)
        # start position should be first base of "unique" ref sequence in variant
        # stop should be first base of ref sequence after variant
        global_starts = variants_global_offset.start.copy()
        global_stops = variants_global_offset.stop.copy()
        between_variants_start = np.insert(global_stops, 0, 0)
        between_variants_end = np.insert(global_starts, len(global_starts), len(global_reference_sequence))
        if not np.all(between_variants_start[1:] >= between_variants_start[:-1]):
            logging.error("Some variants start before others end. Are there overlapping variants?")
            where = np.where(between_variants_start[1:] < between_variants_start[:-1])
            for variant in variants[where]:
                print(f"{variant.position}\t\t{variant.ref_seq}\t\t{variant.alt_seq}")
            raise
        # some variants will start and end at the same position. Then we don't need the sequence between them
        adjust_ends = np.where(between_variants_end < between_variants_start)[0]
        logging.info("Adjusting ends for %d variants because these are before starts", len(adjust_ends))
        # these ends should have starts that match the next start
        assert np.all(between_variants_start[adjust_ends] == between_variants_start[adjust_ends + 1])
        # adjust these to be the same as start
        between_variants_end[adjust_ends] = between_variants_start[adjust_ends]
        #between_variants_end = np.maximum(between_variants_start, between_variants_end)

        sequence_between_variants = bnp.ragged_slice(global_reference_sequence, between_variants_start,
                                                     between_variants_end)
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
