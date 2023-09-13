import bionumpy as bnp
import numpy as np
from bionumpy.bnpdataclass import bnpdataclass

from kage.preprocessing.variants import Variants


@bnpdataclass
class VcfEntryWithSingleIndividualGenotypes:
    chromosome: str
    position: int
    id: str
    ref_seq: str
    alt_seq: str
    quality: str
    filter: str
    info: str
    genotype: str



class VcfWriter:
    def __init__(self, variants: Variants, numeric_genotypes: np.ndarray,
                 sample_name: str = "sample", header: str = None):
        self._variants = variants
        self._numeric_genotypes = numeric_genotypes
        self._sample_name = sample_name
        self._header = header

    def write(self, file_name):
        pass
