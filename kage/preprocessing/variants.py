import dataclasses
import itertools
from dataclasses import dataclass
import npstructures as nps
import awkward as ak
import bionumpy as bnp
import numba
import numpy as np
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy import EncodedRaggedArray, Interval
import logging
from typing import List, Tuple, Union, Iterable
from bionumpy.io.vcf_buffers import VCFBuffer
from bionumpy.streams import NpDataclassStream
from bionumpy.typing import SequenceID

from kage.io import CustomVCFBuffer, SimpleVcfEntry
from kage.util import log_memory_usage_now


@dataclass
class VariantToNodes:
    ref_nodes: np.ndarray
    var_nodes: np.ndarray


@bnpdataclass
class Variants:
    """
    Simple compact representation of variants. Indels are not padded.
    """
    chromosome: SequenceID
    position: int  # position is first ref position in ref sequence or first position after alt path starts (for indels)
    ref_seq: str  # position + len(ref_seq) will always give first ref base after the variant is finished
    alt_seq: str

    def replace_ns(self):
        is_n_ref = self.ref_seq == "N"
        if np.sum(is_n_ref) > 0:
            logging.warning("Some variant alleles on the ref contains Ns. Will treat these as A to be able to compute kmers.")
            self.ref_seq[is_n_ref] = "A"
        is_n_alt = self.alt_seq == "N"
        if np.sum(is_n_alt) > 0:
            logging.warning("Some variant alleles on the alt contains Ns. Will treat these as A to be able to compute kmers.")
            self.alt_seq[is_n_alt] = "A"

    def to_simple_vcf_entry_with_padded_indels(self, reference_genome: bnp.io.indexed_fasta.IndexedFasta):
        """
        Pads indels and adjusts positions (oposite of from_vcf_entry) and returns a SimpleVcfEntry object
        """
        is_indel = (self.ref_seq.shape[1] == 0) | (self.alt_seq.shape[1] == 0)
        pad_size = np.zeros(len(self.position), dtype=int)
        pad_size[is_indel] = 1
        indel_padding_intervals = bnp.Interval(self.chromosome, self.position-1, self.position -1 + pad_size)

        padded_sequence = reference_genome.get_interval_sequences(indel_padding_intervals).raw()

        new_ref = np.concatenate([padded_sequence, self.ref_seq.raw()], axis=1)
        new_ref = bnp.EncodedRaggedArray(bnp.EncodedArray(new_ref.ravel(), self.ref_seq.encoding), new_ref.shape)

        new_alt = np.concatenate([padded_sequence, self.alt_seq.raw()], axis=1)
        new_alt = bnp.EncodedRaggedArray(bnp.EncodedArray(new_alt.ravel(), self.alt_seq.encoding), new_alt.shape)

        return SimpleVcfEntry(self.chromosome, self.position - pad_size, new_ref, new_alt)

    @classmethod
    def from_multiallelic_vcf_entry(cls, variants: Union[bnp.datatypes.VCFEntry, SimpleVcfEntry], return_n_alleles_per_variant=False, remove_indel_padding=True):
        """ Create a Variants object from a multiallelic vcf entry where no variants
        are overlapping (variants are padded).
        Converts all multiallelic variants to biallelic.
        """

        if isinstance(variants, bnp.datatypes.VCFEntry):
            variants = SimpleVcfEntry(variants.chromosome, variants.position, variants.ref_seq, variants.alt_seq)

        # find variants with multiple alleles, split these into multiple variants
        # then call from_vcf_entry
        n_alleles_per_variant = np.sum(variants.alt_seq == ",", axis=1) + 1
        is_multiallelic = n_alleles_per_variant > 1
        logging.info("%d variants are multiallelic" % np.sum(is_multiallelic))

        new = []
        prev_variant = 0
        for i, multiallelic_variant_start in enumerate(np.where(is_multiallelic)[0]):
            variant = variants[multiallelic_variant_start]
            alt_sequences = variant.alt_seq.to_string().split(",")
            biallelic_variants = SimpleVcfEntry.from_entry_tuples([
                (variant.chromosome.to_string(), variant.position, variant.ref_seq.to_string(), alt_sequence)
                for alt_sequence in alt_sequences
            ])
            # add all variants before this
            new.append(variants[prev_variant:multiallelic_variant_start])
            # add the new biallelic created
            new.append(biallelic_variants)
            prev_variant = multiallelic_variant_start + 1

        new.append(variants[prev_variant:])
        merged = np.concatenate(new)
        merged = cls.from_vcf_entry(merged, remove_indel_padding=remove_indel_padding)
        if return_n_alleles_per_variant:
            return merged, n_alleles_per_variant
        return merged

    @classmethod
    def from_vcf_entry(cls, variants: bnp.datatypes.VCFEntry, remove_indel_padding=True):
        variant_ref_sequences = variants.ref_seq
        variant_alt_sequences = variants.alt_seq
        variants_start = variants.position

        if remove_indel_padding:
            logging.info("Removing trailing bases from indels")
            # find indels to remove padding
            is_indel = (variants.ref_seq.shape[1] > 1) | (variants.alt_seq.shape[1] > 1)
            variants_start[is_indel] += 1
            variants_stop = variants_start + variants.ref_seq.shape[1]
            variants_stop[is_indel] -= 1


            # remove padding from ref and alt seq
            mask = np.ones_like(variant_ref_sequences.raw(), dtype=bool)
            mask[is_indel, 0] = False
            variant_ref_sequences = bnp.EncodedRaggedArray(variant_ref_sequences[mask], mask.sum(axis=1))

            mask = np.ones_like(variant_alt_sequences.raw(), dtype=bool)
            mask[is_indel, 0] = False
            variant_alt_sequences = bnp.EncodedRaggedArray(variant_alt_sequences[mask], mask.sum(axis=1))

        #assert np.all(variants_start[1:] >= variants_start[:-1]), "Variants in vcf must be sorted by position within each chromosome, %s" % variants

        return cls(variants.chromosome, variants_start, variant_ref_sequences, variant_alt_sequences)

    def get_multi_allele_variant_alleles(self) -> Tuple['MultiAllelicVariantSequences', 'VariantAlleleToNodeMap', Interval]:
        """
        Groups variants that are exactly overlapping and returns a MultiAllelicVariantSequences object.
        This only makes sense when overlapping variants have been padded first so that they are perfectly
        overlapping.

        Also returns a VariantAlleleToNodeMap object that maps variant id + allele to new variant IDs and alleles.
        """
        sequences = []
        prev_pos = -1
        prev_ref_seq = ""
        prev_chrom = ""
        intervals = []

        for i, variant in enumerate(self):
            chrom = variant.chromosome.to_string()
            ref_seq = variant.ref_seq.to_string()
            alt_seq = variant.alt_seq.to_string()

            # If variants start and end at same position as previous, these are perfectly overlapping
            is_overlapping = False
            if variant.position == prev_pos and chrom == prev_chrom:
                if variant.position + len(ref_seq) == prev_pos + len(prev_ref_seq):
                    is_overlapping = True
                else:
                    if len(prev_ref_seq) != 0:
                        logging.error(variant)
                        logging.error("Two variants are starting at same position, but not ending at same position. These are overlapping and should have been padded")
                        raise Exception("")
            
            #if variant.position == prev_pos and chrom == prev_chrom and variant.position + len(ref_seq):
            if is_overlapping:
                #if ref_seq != self[i-1].ref_seq.to_string():
                #    logging.error(f"Ref seq {ref_seq} does not match previous ref seq {self[i-1].ref_seq.to_string()} on variant {variant}")
                #    raise Exception("Overlapping variants must have same ref sequence")
                sequences[-1].append(alt_seq)
            else:
                sequences.append([ref_seq, alt_seq])
                intervals.append((chrom, variant.position, variant.position + len(ref_seq)))

            prev_pos = variant.position
            prev_chrom = chrom
            prev_ref_seq = ref_seq

        n_alleles_per_variant = [len(s) for s in sequences]

        log_memory_usage_now("Before creating MultiAllelicVariantSequences object from list")
        return MultiAllelicVariantSequences.from_list(sequences), \
            VariantAlleleToNodeMap.from_n_alleles_per_variant(n_alleles_per_variant), \
            Interval.from_entry_tuples(intervals)

    def group_by_chromosome(self):
        """
        Returns an Iterable of Variants-objects by chromosomes. Assumes this variant object is sorted by chromosome
        """
        # Todo: The tolist approach might be slow/memory intensive
        chromosomes = np.array(self.chromosome.tolist())
        splits = np.where(chromosomes[1:] != chromosomes[:-1])[0] + 1
        splits = np.insert(splits, 0, 0)
        splits = np.append(splits, len(chromosomes))
        for start, end in zip(splits[:-1], splits[1:]):
            yield self[start:end]


@dataclass
class VariantAlleleToNodeMap:
    node_ids: nps.RaggedArray  # rows are variant, columns are alleles. Values are node ids
    biallelic_ref_nodes: np.ndarray  # node ids for biallelic ref alleles
    biallelic_alt_nodes: np.ndarray

    """
    Index for looking up variant id and allele => variant id and allele
    """

    def lookup(self, variant_ids, alleles):
        return self.node_ids[variant_ids, alleles]

    @property
    def n_biallelic_variants(self):
        return len(self.biallelic_ref_nodes)

    @property
    def n_nodes(self):
        return self.node_ids.max() + 1

    def get_ref_node(self, variant_id):
        return self.biallelic_ref_nodes[variant_id]

    def get_alt_node(self, variant_id):
        return self.biallelic_alt_nodes[variant_id]

    def get_variant_to_nodes(self):
        return VariantToNodes(self.biallelic_ref_nodes, self.biallelic_alt_nodes)

    def n_alleles_per_variant(self):
        return self.node_ids.shape[1]
    def haplotypes_to_node_ids(self, haplotypes):
        return self.node_ids[np.arange(len(self.node_ids)), haplotypes]

    @property
    def n_alleles_per_variant(self):
        return self.node_ids.shape[1]

    @classmethod
    def from_n_alleles_per_variant(cls, n_alleles_per_variant: List[int]):
        data = np.arange(np.sum(n_alleles_per_variant), dtype=np.int32)
        row_lengths = np.array(n_alleles_per_variant)
        node_ids = nps.RaggedArray(data, row_lengths)

        biallelic_ref_nodes = []
        biallelic_alt_nodes = []
        for multiallelic in node_ids:
            for alt_allele in multiallelic[1:]:
                biallelic_ref_nodes.append(multiallelic[0])
                biallelic_alt_nodes.append(alt_allele)

        return cls(node_ids, biallelic_ref_nodes, biallelic_alt_nodes)


class VariantPadder:
    """
    Merging of overlapping variants into non-overlapping
    variants that start and end at the same position
    """
    def __init__(self, variants: Variants, reference: bnp.EncodedArray):
        assert isinstance(variants, Variants), "Must be Variants object (not VcfEntry or something else)"
        self._variants = variants
        """
        if not np.all(variants.position[1:] >= variants.position[:-1]):
            is_wrong = np.where(variants.position[1:] < variants.position[:-1])[0]
            logging.warning("Variants that are wrong: %s" % variants[1:][is_wrong])
            logging.warning("Prev variants : %s" % variants[:-1][is_wrong])
            raise Exception("Variants must be sorted by position")
        """
        self._reference = reference

    def get_reference_mask(self, threshold=1):
        variants = self._variants
        mask = VariantPadder.get_n_variants_on_ref(variants)
        mask = np.cumsum(mask) >= threshold
        return mask

    @staticmethod
    def get_n_variants_on_ref(variants):
        variants_start = variants.position
        variants_stop = variants_start + variants.ref_seq.shape[1]
        highest_pos = np.max(variants_stop + 2)
        # print("Highest pos", highest_pos)
        mask = np.zeros(highest_pos)
        mask += np.bincount(variants_start, minlength=highest_pos)
        mask -= np.bincount(variants_stop, minlength=highest_pos)
        return mask

    def get_mask_of_consecutive_ref_bases(self, dir='right'):
        # returns number of ref-bases that go from base to next base within variants

        variants_start = self._variants.position
        # variants_stop is the last base of each variant
        variants_stop = variants_start + self._variants.ref_seq.shape[1]-1
        highest_pos = np.max(variants_stop + 2)
        # we don't need variant_starts and stops for insertions, as they don't go over any ref base pairs
        variant_ref_lengths = self._variants.ref_seq.shape[1]
        variants_start = variants_start[variant_ref_lengths > 0]
        variants_stop = variants_stop[variant_ref_lengths > 0]
        # print("Highest pos", highest_pos)

        mask = np.zeros(highest_pos)

        if dir == 'left':
            mask += np.bincount(variants_stop, minlength=highest_pos)
            mask -= np.bincount(variants_start, minlength=highest_pos)
            return (np.cumsum(mask[::-1]) > 0)[::-1]
        else:
            mask += np.bincount(variants_start, minlength=highest_pos)
            mask -= np.bincount(variants_stop, minlength=highest_pos)
            return np.cumsum(mask) > 0

    def get_distance_to_ref_mask(self, dir="left"):
        # for every pos, find number of bases to the end of the region (how much to pad to the side)
        #mask = self.get_reference_mask().astype(int)
        mask = self.get_mask_of_consecutive_ref_bases(dir).astype(int)
        if dir == "left":
            mask = mask[::-1]

        @numba.jit(nopython=True)
        def compute(mask):
            # return a dist-array where each element is number of bases to next
            # 0 in mask
            dist = np.zeros_like(mask)
            # count backwards and reverse
            mask = mask[::-1]
            prev_zero = 0
            for i in range(len(mask)):
                if mask[i] != 0:
                    dist[i] = i - prev_zero
                if mask[i] == 0:
                    prev_zero = i

            return dist[::-1]

        dist = compute(mask)
        if dir == "left":
            return dist[::-1]

        return dist

        starts = np.ediff1d(mask, to_begin=[0]) == 1
        cumsum = np.cumsum(mask)
        cumsum[cumsum >= 1] += 1
        assert np.all(cumsum >= 0)
        mask2 = mask.copy()

        # idea is to subtract the difference of the cumsum at this variant and the previous (what the previous variant increased)
        subtract = cumsum[np.nonzero(starts)]-cumsum[np.insert(np.nonzero(starts), 0, 0)[:-1]]

        mask2[starts] -= (subtract)
        dists = np.cumsum(mask2)

        if not np.all(dists >= 0):
            print("SIDE", dir)
            for v in self._variants:
                print(v.chromosome, v.position, len(v.ref_seq), len(v.alt_seq))
            print(np.where(dists <0), dists[dists < 0])
            assert False
        dists[mask == 0] = 0

        if dir == "right":
            return dists[::-1]

        return dists

    def run(self):
        """
        Pad all variants in overlapping regions so that there are no overlapping variants.
        """
        # find variants that need to be padded
        pad_left = self.get_distance_to_ref_mask(dir="left")
        assert np.all(pad_left >= 0), pad_left[pad_left < 0]

        pad_right = self.get_distance_to_ref_mask(dir="right")
        assert np.all(pad_right >= 0)

        # left padding
        #at_least_two_variants_at_pos = self.get_reference_mask(threshold=2)
        to_pad = pad_left[self._variants.position] > 0
        start_of_padding = self._variants.position[to_pad] - pad_left[self._variants.position[to_pad]]
        end_of_padding = self._variants.position[to_pad]
        left_padding = bnp.ragged_slice(self._reference, start_of_padding, end_of_padding)

        # make new ragged array with the padded sequences
        lengths_left = np.zeros(len(self._variants))
        lengths_left[to_pad] = left_padding.shape[1]
        left_padding = EncodedRaggedArray(left_padding.ravel(), lengths_left)

        # position is adjusted by left padding
        new_positions = self._variants.position.copy()
        new_positions -= lengths_left.astype(int)

        # right padding
        #at_least_two_variants_at_pos = self.get_reference_mask(threshold=2)
        end_pos = self._variants.position + self._variants.ref_seq.shape[1] - 1
        to_pad = (pad_right[end_pos] >= 1) #& (at_least_two_variants_at_pos[end_pos] > 0)

        subset = self._variants[to_pad]
        start_of_padding = subset.position + subset.ref_seq.shape[1] + 0
        end_of_padding = start_of_padding + pad_right[start_of_padding] + 1
        right_padding = bnp.ragged_slice(self._reference, start_of_padding, end_of_padding)



        # make new ragged array with the padded sequences
        lengths_right = np.zeros(len(self._variants))
        lengths_right[to_pad] = right_padding.shape[1]
        right_padding = EncodedRaggedArray(right_padding.ravel(), lengths_right)


        logging.info(f"{np.sum(right_padding.shape[1] > 0)} variants were padded to the right")
        logging.info(f"{np.sum(left_padding.shape[1] > 0)} variants were padded to the left")

        ref_merged = np.concatenate([left_padding.raw(), self._variants.ref_seq.raw(), right_padding.raw()], axis=1)
        alt_merged = np.concatenate([left_padding.raw(), self._variants.alt_seq.raw(), right_padding.raw()], axis=1)
        new_ref_sequences = bnp.EncodedRaggedArray(bnp.EncodedArray(ref_merged.ravel(), bnp.BaseEncoding), ref_merged.shape)
        new_alt_sequences = bnp.EncodedRaggedArray(bnp.EncodedArray(alt_merged.ravel(), bnp.BaseEncoding), alt_merged.shape)

        return Variants(self._variants.chromosome, new_positions, new_ref_sequences, new_alt_sequences)


def get_padded_variants_from_vcf(variants: 'VariantStream', reference_file_name, also_return_original_variants=False,
                                 remove_indel_padding=True,
                                 remove_sequence_from_low_af_deletions: float = 0.0
                                 ) -> Variants:
    if isinstance(variants, str):
        variants = VariantStream.from_vcf(variants)

    genome = bnp.open(reference_file_name).read()
    sequences = {str(sequence.name): sequence.sequence for sequence in genome}
    all_variants = []
    all_vcf_variants = []
    n_alleles_per_variant = []

    for chromosome, raw_chromosome_variants in variants.read_by_chromosome():  #bnp.groupby(variants, "chromosome"):
        assert chromosome in sequences, ("Chromosome %s not found in reference genome. "
                                         "Check that reference genome contains all variants in VCF.") % chromosome
        if also_return_original_variants:
            r = raw_chromosome_variants
            all_vcf_variants.append(SimpleVcfEntry(r.chromosome, r.position, r.ref_seq.copy(), r.alt_seq.copy()))

        if remove_sequence_from_low_af_deletions > 0:
            raw_chromosome_variants = remove_alt_and_ref_seq_on_deletions_with_low_af(raw_chromosome_variants, remove_sequence_from_low_af_deletions)

        n_alleles_per_variant.append(np.sum(raw_chromosome_variants.alt_seq == ",", axis=1) + 2)
        chromosome_variants = Variants.from_multiallelic_vcf_entry(raw_chromosome_variants, remove_indel_padding=remove_indel_padding)
        log_memory_usage_now("After creating Variants")
        logging.info("Padding variants on chromosome " + chromosome)
        logging.info("%d variants" % len(chromosome_variants))

        padded_variants = VariantPadder(chromosome_variants, sequences[chromosome]).run()
        log_memory_usage_now("After padding")
        all_variants.append(padded_variants)

        log_memory_usage_now("After processing variants for chromosome %s" % chromosome)

    all_variants = np.concatenate(all_variants)
    logging.info(f"In total {len(all_variants)} variants")
    if also_return_original_variants:
        return all_variants, np.concatenate(all_vcf_variants), np.concatenate(n_alleles_per_variant)

    return all_variants


def pad_vcf_cli(args):
    variants = get_padded_variants_from_vcf(args.vcf_file_name, args.reference)
    original_variants = bnp.open(args.vcf_file_name).read_chunks()
    offset = 0
    with bnp.open(args.out_file_name, "w") as out:
        for chunk in original_variants:
            variant_chunk = variants[offset:offset + len(chunk)]
            chunk.position = variant_chunk.position
            chunk.ref_j

            offset += len(chunk)


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

    @property
    def variant_sizes(self):
        """
        Return size of ref allele of each variant.
        """
        return ak.to_numpy(ak.num(self._data[:, 0]))

    @classmethod
    def from_list(cls, variant_sequences: List[List]):
        return cls(ak.Array(variant_sequences))

    def get_haplotype_sequence(self, haplotypes: np.ndarray, from_variant: int = None, to_variant: int = None) -> bnp.EncodedRaggedArray:
        assert len(haplotypes) == self.n_variants
        if from_variant is None:
            from_variant = 0
        if to_variant is None:
            to_variant = len(haplotypes)

        allele_sequences = self._data[np.arange(from_variant, to_variant), haplotypes[from_variant:to_variant]]
        shape = ak.to_numpy(ak.num(allele_sequences))
        # flatten, and encode
        bytes = ak.to_numpy(ak.flatten(ak.without_parameters(allele_sequences)))
        return bnp.EncodedRaggedArray(bnp.EncodedArray(bytes, bnp.BaseEncoding), shape)
        #return bnp.EncodedRaggedArray(bnp.change_encoding(bnp.EncodedArray(bytes, bnp.BaseEncoding), bnp.DNAEncoding), shape)

    def get_haplotype_sequence_at_variant(self, variant, haplotype) -> bnp.EncodedArray:
        sequence = ak.to_numpy(self._data[variant, haplotype])
        return bnp.EncodedArray(bnp.change_encoding(bnp.EncodedArray(sequence, bnp.BaseEncoding), bnp.DNAEncoding), bnp.DNAEncoding)

    def to_list(self):
        return ak.to_list(self._data)


class VariantStream:
    """
    Wrapper around bnp's read_chunks() on a vcf.
    """
    def __init__(self, stream: Iterable):
        self._stream = stream

    @classmethod
    def from_vcf(cls, vcf_file_name, buffer_type=VCFBuffer, min_chunk_size=50000000):
        logging.info("Using buffer type %s" % buffer_type)
        chunks = bnp.open(vcf_file_name, buffer_type=buffer_type).read_chunks(min_chunk_size)
        return cls(chunks)

    def _read_chunks(self) -> Iterable:
        return self._stream

    def read_chunks(self) -> NpDataclassStream:
        return NpDataclassStream(self._read_chunks())

    def read_by_chromosome(self):
        return bnp.groupby(self.read_chunks(), "chromosome")

    def raw(self):
        return self._stream


class FilteredVariantStream(VariantStream):
    """Subclass of VariantStream with a mask of what to keep.
    Will only give entries that are to be kept according to mask"""
    def __init__(self, stream: NpDataclassStream, to_keep: np.ndarray):
        self._stream = stream
        self._to_keep = to_keep

    def _read_chunks(self):
        prev = 0
        for chunk in self._stream:
            logging.info(f"{len(chunk)}, {prev}, {prev+len(chunk)}")
            assert prev + len(chunk) <= len(self._to_keep)
            yield chunk[self._to_keep[prev:prev+len(chunk)]]
            prev += len(chunk)

        #return (chunk[mask] for chunk, mask in zip(self._stream.read_chunks(), self._to_keep))

    @classmethod
    def from_vcf_with_snps_indels_inside_svs_removed(cls, vcf_file_name, sv_size_limit=50,
                                                     buffer_type=bnp.io.vcf_buffers.VCFBuffer,
                                                     min_chunk_size=10000000,
                                                     filter_using_other_vcf=None):
        to_keep = []
        vcf = filter_using_other_vcf if filter_using_other_vcf is not None else vcf_file_name
        stream = VariantStream.from_vcf(vcf, buffer_type=CustomVCFBuffer)  # only need some simple VCFBuffer for filtering
        #for chromosome, chunk in stream.read_by_chromosome():
        for chunk in stream.read_chunks():
            to_keep.append(~find_snps_indels_covered_by_svs(chunk, sv_size_limit=sv_size_limit, allow_approx=True))

        to_keep = np.concatenate(to_keep)
        logging.info(f"{np.sum(~to_keep)} snps/indels inside SVs filtered out")
        logging.info(f"{np.sum(to_keep)} variants kept")
        return cls(VariantStream.from_vcf(vcf_file_name, buffer_type=buffer_type, min_chunk_size=min_chunk_size).raw(), to_keep)


class FilteredOnMaxAllelesVariantStream(VariantStream):
    """
    Filters variants by max alleles. Only works for multiallelic variants (since
    then number of alleles is known)
    """
    def __init__(self, stream: VariantStream, max_alleles: int):
        self._stream = stream
        self._max_alleles = max_alleles

    def _read_chunks(self):
        for chunk in self._stream.read_chunks():
            mask = np.sum(chunk.alt_seq == ",", axis=1) + 2 <= self._max_alleles
            logging.info("Filtering away %d variants because more than %d alleles" % (np.sum(~mask), self._max_alleles))
            yield chunk[mask]


class LowAfDeletionsReplacedVariantStream(VariantStream):
    """
    Replaces alleles on low-af deletions with empty sequences
    """
    def __init__(self, stream: VariantStream, min_af):
        self._stream = stream
        self._min_af = min_af

    def _read_chunks(self):
        n_filtered = 0
        for chunk in self._stream.read_chunks():
            #allele_frequencies = get_af_from_info_string_in_vcf_chunk(chunk)
            filter_out = self.get_filter_of_deletions_with_low_af(chunk, self._min_af)
            n_filtered += np.sum(filter_out)
            logging.info(f"Filtered out {np.sum(filter_out)} deletions with AF < {self._min_af}. In total filtered {n_filtered}")
            yield chunk[~filter_out]
            continue


def get_filter_of_deletions_with_low_af_only_deletions_overlapping_other_variants(chunk, min_af):
    variant_ref_sizes = chunk.ref_seq.shape[1]
    #max_ref_pos = np.max(chunk.position + variant_ref_sizes)
    #n_variants_on_ref_mask = np.zeros(max_ref_pos+1)
    #for i, variant in enumerate(chunk):
    #    n_variants_on_ref_mask[int(variant.position):int(variant.position)+variant_ref_sizes[i]] += 1
    large_variants = chunk[(chunk.ref_seq.shape[1] > 10) | (chunk.alt_seq.shape[1] > 10)]
    filter = np.zeros(len(chunk), dtype=bool)
    if len(large_variants) == 0:
        # no large variants, no need to filter
        return filter

    n_variants_on_ref_mask = VariantPadder.get_n_variants_on_ref(large_variants)

    for deletion in np.nonzero(get_filter_of_deletions_with_low_af(chunk, min_af, min_size=20))[0]:
        if np.any(n_variants_on_ref_mask[
                  int(chunk.position[deletion]):int(chunk.position[deletion])+variant_ref_sizes[deletion]
                  ] > 1):
            filter[deletion] = True

    return filter


def remove_alt_and_ref_seq_on_deletions_with_low_af(chunk, min_af):
    #filter_out = get_filter_of_deletions_with_low_af(chunk, min_af)
    filter_out = get_filter_of_deletions_with_low_af_only_deletions_overlapping_other_variants(chunk, min_af)
    logging.info(f"{np.sum(filter_out)} deletions with AF < {min_af} will not be included in padding/kmer indexing")
    # Set ref seq and alt seq to "" for these variants
    new_ref = bnp.as_encoded_array([
        seq.to_string()[0] if filter_out[i] else seq.to_string() for i, seq in enumerate(chunk.ref_seq)
    ])
    new_alt = bnp.as_encoded_array([
        seq.to_string()[0] if filter_out[i] else seq.to_string() for i, seq in enumerate(chunk.alt_seq)
    ])
    chunk.ref_seq = new_ref
    chunk.alt_seq = new_alt
    return chunk


def get_filter_of_deletions_with_low_af(chunk, min_af, min_size=20):
    allele_frequencies = chunk.info.AF[:, 0]
    #allele_frequencies = bnp.io.strops.str_to_float(allele_frequencies)
    filter_out = (allele_frequencies < min_af) & (chunk.ref_seq.shape[1] >= min_size)  # "big" deletions
    return filter_out


class FilteredOnMaxAllelesVariantStream2(VariantStream):
    """Another version that uses a counter on the references.
    Works for biallelic.
    The filtering is approximate and may miss variants on chunk borders.
    """
    def __init__(self, stream: VariantStream, max_alleles: int):
        self._stream = stream
        self._max_alleles = max_alleles

    def _read_chunks(self):
        for chunk in self._stream.read_chunks():
            # Note: This does not adjust for the fact that indels have
            # a trailing base. So not perfectly accuracey, but probably "good enough"
            # May filter a bit more than necessary

            starts = chunk.position
            ends = starts + chunk.ref_seq.shape[1]

            first_variant_start = np.min(starts)
            last_end = np.max(ends)

            starts = starts - first_variant_start
            ends = ends - first_variant_start

            size = last_end-first_variant_start
            allele_counter = np.zeros(size, dtype=int)

            @numba.jit(nopython=True)
            def _get_counts(starts, ends, counter):
                for i in range(len(starts)):
                    counter[starts[i]:ends[i]] += 1

            _get_counts(starts, ends, allele_counter)

            # find which variants to filter out
            @numba.jit(nopython=True)
            def _get_filter(filter, starts, ends, counter, max_alleles):
                for i in range(len(starts)):
                    if np.max(counter[starts[i]:ends[i]]) > max_alleles:
                        filter[i] = True
                return filter

            filter = np.zeros(len(starts), dtype=bool)
            _get_filter(filter, starts, ends, allele_counter, self._max_alleles)

            logging.info("Filtering away %d variants because more than %d alleles" % (np.sum(filter), self._max_alleles))
            yield chunk[~filter]


def filter_variants_with_more_alleles_than(biallelic_padded_variants: Variants, original_vcf_variants, n_alleles_per_original_variant, max_alleles: int):

    filters = []
    for chromosome_chunk in biallelic_padded_variants.group_by_chromosome():
        start_positions = chromosome_chunk.position
        n_overlapping = np.bincount(start_positions)

        filter = n_overlapping[start_positions] > max_alleles
        filters.append(filter)

    filter = np.concatenate(filters)
    logging.info(f"{np.sum(filter)} variants overlap so that they end up with more than {max_alleles} alleles. These will be ignored")

    # Find out which original vcf variants these filtered correspond to
    mask = nps.RaggedArray(filter, n_alleles_per_original_variant-1)
    filter_original = np.any(mask, axis=1)

    return biallelic_padded_variants[~filter], original_vcf_variants[~filter_original], n_alleles_per_original_variant[~filter_original], filter_original


class VariantStreamWithoutVariantsWithSymbolicAlleles(VariantStream):
    """
    Does not give variants where alt allele is *
    """
    def _read_chunks(self):
        for chunk in self._stream:
            mask = np.all(chunk.alt_seq != "*", axis=1)
            yield chunk[mask]


def find_snps_indels_covered_by_svs(variants: bnp.datatypes.VCFEntry, sv_size_limit: int = 50, allow_approx=False) -> np.ndarray:
    """
    Returns a boolean mask where True are SNPs/indels that are covered by a SV.
    Assumes all variants are on the same chromosome.
    """
    if not allow_approx:
        assert variants.chromosome[0].to_string() == variants.chromosome[-1].to_string()
    is_snp_indel = (variants.ref_seq.shape[1] <= sv_size_limit) & (variants.alt_seq.shape[1] <= sv_size_limit)
    is_sv = ~is_snp_indel

    is_any_indel = (variants.ref_seq.shape[1] > 1) | (variants.alt_seq.shape[1] > 1)
    starts = variants.position
    starts[is_any_indel] += 1  # indels are padded with one base
    ends = starts + variants.ref_seq.shape[1] - 1

    sv_position_mask = np.zeros(np.max(ends) + 1, dtype=bool)
    indexes_of_covered_by_sv = nps.ragged_slice(np.arange(len(sv_position_mask)), starts[is_sv], ends[is_sv]).ravel()
    sv_position_mask[indexes_of_covered_by_sv] = True

    is_covered = (sv_position_mask[starts] | sv_position_mask[ends]) & is_snp_indel

    return is_covered


def find_end_in_info_string(info_string: str) -> int:
    fields = info_string.split(";")
    fields = [f for f in fields if f.startswith("END=")]
    assert len(fields) == 1
    value = fields[0].split("=")[1]
    return int(value)


def get_af_from_info_string_in_vcf_chunk(chunk) -> np.ndarray:
    # hacky way to get allele frequency fast
    info_fields = bnp.io.strops.join(chunk.info, ";")
    info_fields = bnp.io.strops.split(info_fields, sep=";")
    af_strings = info_fields[bnp.str_equal(info_fields[:, 0:3], "AF=")]
    assert len(af_strings) == len(chunk), "Not all lines contain AF=?"

    # afs = np.array([float(af_string.to_string()) for af_string in af_strings[:, 3:]])
    afs = (af_string.to_string() for af_string in af_strings[:, 3:])
    afs = nps.RaggedArray(
        [
            [float(af) for af in af_string.split(",")]
            for af_string in afs
        ]
    )
    # use highest af
    afs = np.max(afs, axis=-1)
    return afs


def convert_purebread_vcf(vcf_file_name: str, out_file_name: str):
    purebread = bnp.open(vcf_file_name, buffer_type=bnp.io.vcf_buffers.VCFBuffer2).read_chunks()
    out = bnp.open(out_file_name, "w", buffer_type=bnp.io.vcf_buffers.VCFBuffer2)

    lookup = {"0": "0|0", "1": "1|1", ".": ".|."}
    for i, chunk in enumerate(purebread):
        logging.info(f"Converting chunk {i}")
        g = chunk.genotype.raw()
        try:
            new = bnp.string_array.string_array([[lookup[genotype.decode("utf-8")] for genotype in row] for row in g])
        except KeyError:
            logging.info("Some genptypes not found in lookup")
            print(chunk.genotype)
            raise

        chunk = bnp.replace(chunk, genotype=new)
        out.write(chunk)

    logging.info("Done converting")



