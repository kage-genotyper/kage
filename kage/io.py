import bionumpy as bnp
import numpy as np
from bionumpy.bnpdataclass import bnpdataclass

from kage.preprocessing.variants import Variants, SimpleVcfEntry


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



def write_vcf(variants: SimpleVcfEntry, numeric_genotypes: np.ndarray,
                 sample_name: str = "sample", header: str = None):

    genotypes = [str(g) for g in numeric_genotypes]
    entry = VcfEntryWithSingleIndividualGenotypes(
        variants.chromosome,
        variants.position,
        [str(i) for i in range(len(variants))],
        variants.ref_seq,
        variants.alt_seq,
        ["." for i in range(len(variants))],
        ["." for i in range(len(variants))],
        ["." for i in range(len(variants))],
        genotypes
    )
