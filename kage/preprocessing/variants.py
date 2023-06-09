import bionumpy as bnp
import numpy as np


class VariantMerger:
    """
    Merging of overlapping variants into non-overlapping
    variants that start and end at the same position
    """
    def __init__(self, variants: bnp.datatypes.VCFEntry, reference: bnp.EncodedArray):
        self._variants = variants
        self._reference = reference

    def get_reference_mask(self):
        print(self._variants)
        not_snp = (self._variants.ref_seq.shape[1] > 1) | (self._variants.alt_seq.shape[1] > 1)
        print(not_snp)
        variants_start = self._variants.position
        variants_start[not_snp] += 1
        variants_stop = variants_start + self._variants.ref_seq.shape[1]
        variants_stop[not_snp] -= 1

        highest_pos = np.max(variants_stop+1)
        mask = np.zeros(highest_pos)
        mask += np.bincount(variants_start, minlength=highest_pos)
        mask -= np.bincount(variants_stop, minlength=highest_pos)
        print(mask)
        mask = np.cumsum(mask) > 0
        return mask


    def merge(self):
        """
        Creat a mask with all ref-bases covered by a variant.
        All variants within masked areas will be expanded.
        """
        mask = self.get_reference_mask()
        print(mask)