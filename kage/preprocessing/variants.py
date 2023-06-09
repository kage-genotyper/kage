import bionumpy as bnp
import numpy as np
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy import EncodedRaggedArray


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

class VariantPadder:
    """
    Merging of overlapping variants into non-overlapping
    variants that start and end at the same position
    """
    def __init__(self, variants: bnp.datatypes.VCFEntry, reference: bnp.EncodedArray):
        self._variants = variants
        self._reference = reference

    def get_reference_mask(self):
        variants_start = self._variants.position
        variants_stop = variants_start + self._variants.ref_seq.shape[1]
        highest_pos = np.max(variants_stop+1)

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

        starts = np.ediff1d(mask, to_begin=[0]) == 1
        cumsum = np.cumsum(mask)
        mask2 = mask.copy()
        mask2[starts] -= cumsum[starts] - 1
        dists = np.cumsum(mask2)
        dists[mask == 0] = 0

        if dir == "right":
            return dists[::-1]

        return dists


    def run(self):
        """
        Pad all variants in overlapping regions so that there are no overlapping variants.
        """
        mask = self.get_reference_mask()

        # find variants that need to be padded
        pad_left = self.get_distance_to_ref_mask(dir="left")
        pad_right = self.get_distance_to_ref_mask(dir="right")

        print("Pad left:", pad_left)
        print("Pad right:", pad_right)

        # left padding
        to_pad = pad_left[self._variants.position] > 1
        start_of_padding = self._variants.position[to_pad] - pad_left[self._variants.position[to_pad]] + 1
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
        to_pad = pad_right[self._variants.position + self._variants.ref_seq.shape[1]] >= 1
        print(to_pad)
        subset = self._variants[to_pad]
        print("Subset position")
        print(subset.position)
        start_of_padding = subset.position + subset.ref_seq.shape[1] + 0
        print("Start of padding")
        print(start_of_padding)
        end_of_padding = start_of_padding + pad_right[start_of_padding] + 0
        print(start_of_padding, end_of_padding)
        right_padding = bnp.ragged_slice(self._reference, start_of_padding, end_of_padding)


        # make new ragged array with the padded sequences
        lengths_right = np.zeros(len(self._variants))
        lengths_right[to_pad] = right_padding.shape[1]
        right_padding = EncodedRaggedArray(right_padding.ravel(), lengths_right)

        print("Padding left")
        print(left_padding)
        print("Padding right")
        print(right_padding)

        lengths_right= np.zeros(len(self._variants))

        print(right_padding.raw())
        ref_merged = np.concatenate([left_padding.raw(), self._variants.ref_seq.raw(), right_padding.raw()], axis=1)
        alt_merged = np.concatenate([left_padding.raw(), self._variants.alt_seq.raw(), right_padding.raw()], axis=1)
        new_ref_sequences = bnp.EncodedRaggedArray(bnp.EncodedArray(ref_merged.ravel(), bnp.BaseEncoding), ref_merged.shape)
        new_alt_sequences = bnp.EncodedRaggedArray(bnp.EncodedArray(alt_merged.ravel(), bnp.BaseEncoding), alt_merged.shape)


        return Variants(self._variants.chromosome, new_positions, new_ref_sequences, new_alt_sequences)
