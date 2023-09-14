from collections import defaultdict

import bionumpy as bnp
import numpy as np
from bionumpy.bnpdataclass import bnpdataclass
import npstructures as nps
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
    format: str
    genotype: str


class VcfWithSingleIndividualBuffer(bnp.io.delimited_buffers.VCFBuffer):
    dataclass = VcfEntryWithSingleIndividualGenotypes


def write_multiallelic_vcf_with_biallelic_numeric_genotypes(variants: SimpleVcfEntry, numeric_genotypes: np.ndarray,
                                                            out_file_name: str, n_alleles_per_variant: np.ndarray,
                                                            sample_name: str = "sample", header: str = ""):
    assert len(n_alleles_per_variant) == len(variants)
    print("N numeric genotypes", len(numeric_genotypes))
    string_genotypes = convert_biallelic_numeric_genotypes_to_multialellic_string_genotypes(n_alleles_per_variant, numeric_genotypes)
    print("N string genotypes: ", len(string_genotypes))
    print(string_genotypes)
    print("N variants:", len(variants))
    write_vcf(variants, string_genotypes, out_file_name, sample_name, header)


def write_vcf(variants: SimpleVcfEntry, string_genotypes: bnp.EncodedRaggedArray,
              out_file_name: str, sample_name: str = "sample", header: str = ""):
    """Numeric genotypes: 1: 0/0, 2: 1/1, 3: 0/1 """

    #string_genotypes = ["0/0", "1/1", "0/1"]
    genotypes = string_genotypes  # [string_genotypes[g-1] for g in numeric_genotypes]

    entry = VcfEntryWithSingleIndividualGenotypes(
        variants.chromosome,
        variants.position,
        ["variant" + str(i) for i in range(len(variants))],
        variants.ref_seq,
        variants.alt_seq,
        ["." for i in range(len(variants))],
        ["." for i in range(len(variants))],
        ["." for i in range(len(variants))],
        ["GT" for i in range(len(variants))],
        genotypes
    )

    entry.set_context("header", header)
    with bnp.open(out_file_name, "w", VcfWithSingleIndividualBuffer) as f:
        f.write(entry)


def create_vcf_header_with_sample_name(existing_vcf_header, sample_name):
    header_lines = existing_vcf_header.split("\n")
    for i, line in enumerate(header_lines):
        if line.startswith("#CHROM"):
            header_lines[i] = '\t'.join(line.split("\t")[0:9]) + "\t" + sample_name
    return "\n".join(header_lines)


def convert_biallelic_numeric_genotypes_to_multialellic_string_genotypes(n_alleles_per_variant: np.ndarray, genotypes: np.ndarray) -> bnp.EncodedRaggedArray:
    """
    Converts genotypes on biallelic variants to multiallelic given the number of
    alleles per variant in a set of multiallelic variants.

    Numeric input genotypes are:
    1: 0/0, 2: 1/1, 3: 0/1

    Output genotypes are strings on the form
    x/y
    These strings are placed in an EncodedRaggedArray
    """
    string_genotypes = ["0/0", "1/1", "0/1"]
    assert np.sum(n_alleles_per_variant - 1) == len(genotypes)

    #if np.all(n_alleles_per_variant == 2):
    #    # all alleles are biallelic
    #    return genotypes

    # group into multiallelic
    multialellic_genotypes = nps.RaggedArray(genotypes, n_alleles_per_variant-1)
    print("N multiallelic variants: ", len(multialellic_genotypes))

    # each row should be converted into a multiallelic genotype
    out = []
    for variant in multialellic_genotypes:
        genotype = "0/0"
        if len(variant) == 1:
            genotype = string_genotypes[variant[0]-1]
        else:
            # multiallelic
            for i, allele in enumerate(variant):
                if allele == 2:
                    genotype = f"{i+1}/{i+1}"
                    break
                elif allele == 3:
                    genotype = f"0/{i+1}"
                    break

        out.append(genotype)

    assert len(out) == len(multialellic_genotypes)
    return bnp.as_encoded_array(out)




