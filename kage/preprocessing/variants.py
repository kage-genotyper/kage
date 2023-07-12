import itertools
from dataclasses import dataclass
import npstructures as nps
import awkward as ak
import bionumpy as bnp
import numpy as np
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy import EncodedRaggedArray, Interval
import logging
from typing import List, Tuple


@dataclass
class VariantToNodes:
    ref_nodes: np.ndarray
    var_nodes: np.ndarray


@bnpdataclass
class Variants:
    """
    Simple compact representation of variants. Indels are not padded.
    """
    chromosome: str
    position: int  # position is first ref position in ref sequence or first position after alt path starts (for indels)
    ref_seq: str  # position + len(ref_seq) will always give first ref base after the variant is finished
    alt_seq: str

    @classmethod
    def from_vcf_entry(cls, variants: bnp.datatypes.VCFEntry):
        # find indels to remove padding
        is_indel = (variants.ref_seq.shape[1] > 1) | (variants.alt_seq.shape[1] > 1)
        variants_start = variants.position
        variants_start[is_indel] += 1
        variants_stop = variants_start + variants.ref_seq.shape[1]
        variants_stop[is_indel] -= 1

        variant_ref_sequences = variants.ref_seq
        variant_alt_sequences = variants.alt_seq

        # remove padding from ref and alt seq
        mask = np.ones_like(variant_ref_sequences.raw(), dtype=bool)
        mask[is_indel, 0] = False
        variant_ref_sequences = bnp.EncodedRaggedArray(variant_ref_sequences[mask], mask.sum(axis=1))

        mask = np.ones_like(variant_alt_sequences.raw(), dtype=bool)
        mask[is_indel, 0] = False
        variant_alt_sequences = bnp.EncodedRaggedArray(variant_alt_sequences[mask], mask.sum(axis=1))

        return cls(variants.chromosome, variants_start, variant_ref_sequences, variant_alt_sequences)

    def to_vcf_entry(self, ):
        # adds trailing bases to indels and adjusts positions
        is_indel = (self.ref_seq.shape[1] != 1) | (self.alt_seq.shape[1] != 1)
        new_positions = self.position.copy()

    def get_multi_allele_variant_alleles(self) -> Tuple['MultiAllelicVariantSequences', 'VariantAlleleToNodeMap', Interval]:
        """
        Groups variants that are exactly overlapping and returns a MultiAllelicVariantSequences object.
        This only makes sense when overlapping variants have been padded first so that they are perfectly
        overlapping.

        Also returns a BiallelicToMultiallelicIndex object that maps variant id + allele to new variant IDs and alleles.
        """
        sequences = []
        prev_pos = -1
        prev_chrom = ""
        intervals = []

        for i, variant in enumerate(self):
            chrom = variant.chromosome.to_string()
            ref_seq = variant.ref_seq.to_string()
            alt_seq = variant.alt_seq.to_string()

            if variant.position == prev_pos and chrom == prev_chrom:
                assert ref_seq == self[i-1].ref_seq.to_string(), "Overlapping variants must have same ref sequence"
                sequences[-1].append(alt_seq)
            else:
                sequences.append([ref_seq, alt_seq])
                intervals.append((chrom, variant.position, variant.position + len(ref_seq)))

            prev_pos = variant.position
            prev_chrom = chrom

        n_alleles_per_variant = [len(s) for s in sequences]

        return MultiAllelicVariantSequences.from_list(sequences), \
            VariantAlleleToNodeMap.from_n_alleles_per_variant(n_alleles_per_variant), \
            Interval.from_entry_tuples(intervals)


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

    def get_variant_to_nodes(self):
        return VariantToNodes(self.biallelic_ref_nodes, self.biallelic_alt_nodes)

    def n_alleles_per_variant(self):
        return self.node_ids.shape[1]

    def haplotypes_to_node_ids(self, haplotypes):
        return self.node_ids[np.arange(len(self.node_ids)), haplotypes]

    @classmethod
    def from_n_alleles_per_variant(cls, n_alleles_per_variant: List[int]):
        data = np.arange(np.sum(n_alleles_per_variant))
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
    def __init__(self, variants: bnp.datatypes.VCFEntry, reference: bnp.EncodedArray):
        assert isinstance(variants, Variants), "Must be Variants object (not VcfEntry or something else)"
        self._variants = variants
        assert np.all(variants.position[1:] >= variants.position[:-1]), "Variants must be sorted by position"
        self._reference = reference

    def get_reference_mask(self):
        variants_start = self._variants.position
        variants_stop = variants_start + self._variants.ref_seq.shape[1]
        highest_pos = np.max(variants_stop+1)
        #print("Highest pos", highest_pos)

        mask = np.zeros(highest_pos)
        mask += np.bincount(variants_start, minlength=highest_pos)
        mask -= np.bincount(variants_stop, minlength=highest_pos)
        mask = np.cumsum(mask) > 0
        return mask

    def get_distance_to_ref_mask(self, dir="left"):
        # for every pos, find number of bases to the end of the region (how much to pad to the side)
        mask = self.get_reference_mask().astype(int)
        if dir == "right":
            mask = mask[::-1]

        #print("Original mask")
        #print(mask)

        starts = np.ediff1d(mask, to_begin=[0]) == 1
        #print(np.nonzero(starts))
        cumsum = np.cumsum(mask)
        #print("CUMSUM")
        #print(cumsum)
        assert np.all(cumsum >= 0)
        mask2 = mask.copy()

        # idea is to subtract the difference of the cumsum at this variant and the previous (what the previous variant increased)
        subtract = cumsum[np.nonzero(starts)]-cumsum[np.insert(np.nonzero(starts), 0, 0)[:-1]]
        #print("Starts")
        #print(np.nonzero(starts))
        #print("SUbtract")
        #print(subtract)
        #print(cumsum[starts])
        mask2[starts] -= (subtract)
        #print("MASK 2 after minus")
        #print(mask2)
        dists = np.cumsum(mask2)
        #print("Dists after cumsum of mask2")
        #print(dists)

        if not np.all(dists >= 0):
            print("SIDE", dir)
            for v in self._variants:
                print(v.chromosome, v.position, len(v.ref_seq), len(v.alt_seq))
            print(np.where(dists <0), dists[dists < 0])
            assert False
        dists[mask == 0] = 0

        #print("Final dists")
        #print(dists)

        if dir == "right":
            return dists[::-1]

        return dists

    def run(self):
        """
        Pad all variants in overlapping regions so that there are no overlapping variants.
        """
        # find variants that need to be padded
        pad_left = self.get_distance_to_ref_mask(dir="left")
        #print()
        #print("DISTS left")
        #print(pad_left)
        assert np.all(pad_left >= 0), pad_left[pad_left < 0]

        pad_right = self.get_distance_to_ref_mask(dir="right")
        #print()
        #print("DISTS RIGHT")
        #print(pad_right)
        assert np.all(pad_right >= 0)

        # left padding
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
        to_pad = pad_right[self._variants.position + self._variants.ref_seq.shape[1] - 1] >= 1

        #print("TO pad right")
        #print(to_pad)
        #print(self._variants.position + self._variants.ref_seq.shape[1])

        subset = self._variants[to_pad]
        #print("Subset position")
        #print(subset.position)
        start_of_padding = subset.position + subset.ref_seq.shape[1] + 0
        #print("Start of padding")
        #print(start_of_padding)
        end_of_padding = start_of_padding + pad_right[start_of_padding] + 1
        #print(start_of_padding, end_of_padding)
        right_padding = bnp.ragged_slice(self._reference, start_of_padding, end_of_padding)



        # make new ragged array with the padded sequences
        lengths_right = np.zeros(len(self._variants))
        lengths_right[to_pad] = right_padding.shape[1]
        right_padding = EncodedRaggedArray(right_padding.ravel(), lengths_right)

        #print("Left padding")
        #print(left_padding[left_padding.shape[1] > 0])

        #print("Right padding")
        #print(right_padding[right_padding.shape[1] > 0])

        logging.info(f"{np.sum(right_padding.shape[1] > 0)} variants were padded to the right")
        logging.info(f"{np.sum(left_padding.shape[1] > 0)} variants were padded to the left")


        #print("Padding left")
        #print(left_padding)
        #print("Padding right")
        #print(right_padding)

        #print(right_padding.raw())
        ref_merged = np.concatenate([left_padding.raw(), self._variants.ref_seq.raw(), right_padding.raw()], axis=1)
        alt_merged = np.concatenate([left_padding.raw(), self._variants.alt_seq.raw(), right_padding.raw()], axis=1)
        new_ref_sequences = bnp.EncodedRaggedArray(bnp.EncodedArray(ref_merged.ravel(), bnp.BaseEncoding), ref_merged.shape)
        new_alt_sequences = bnp.EncodedRaggedArray(bnp.EncodedArray(alt_merged.ravel(), bnp.BaseEncoding), alt_merged.shape)

        return Variants(self._variants.chromosome, new_positions, new_ref_sequences, new_alt_sequences)


def get_padded_variants_from_vcf(vcf_file_name, reference_file_name):
    variants = bnp.open(vcf_file_name).read_chunks()
    genome = bnp.open(reference_file_name).read()
    sequences = {str(sequence.name): sequence.sequence for sequence in genome}
    all_variants = []

    for chromosome, chromosome_variants in bnp.groupby(variants, "chromosome"):
        chromosome_variants = Variants.from_vcf_entry(chromosome_variants)
        logging.info("Padding variants on chromosome " + chromosome)
        logging.info("%d variants" % len(chromosome_variants))
        padded_variants = VariantPadder(chromosome_variants, sequences[chromosome]).run()
        all_variants.append(padded_variants)

    all_variants = np.concatenate(all_variants)
    logging.info(f"In total {len(all_variants)} variants")
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

    def get_haplotype_sequence(self, haplotypes: np.ndarray) -> bnp.EncodedRaggedArray:
        assert len(haplotypes) == self.n_variants
        allele_sequences = self._data[np.arange(self.n_variants), haplotypes]
        shape = ak.to_numpy(ak.num(allele_sequences))
        # flatten, and encode
        bytes = ak.to_numpy(ak.flatten(ak.without_parameters(allele_sequences)))
        return bnp.EncodedRaggedArray(bnp.change_encoding(bnp.EncodedArray(bytes, bnp.BaseEncoding), bnp.DNAEncoding), shape)

    def to_list(self):
        return ak.to_list(self._data)
