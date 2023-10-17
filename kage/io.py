import logging
from collections import defaultdict
from typing import Optional

import bionumpy as bnp
import numpy as np
from bionumpy.bnpdataclass import bnpdataclass
import npstructures as nps
from kage.preprocessing.variants import Variants, SimpleVcfEntry
from kage.util import vcf_pl_and_gl_header_lines


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


class VcfWithSingleIndividualBuffer(bnp.io.VCFBuffer):
    dataclass = VcfEntryWithSingleIndividualGenotypes


def write_multiallelic_vcf_with_biallelic_numeric_genotypes(variants: SimpleVcfEntry, numeric_genotypes: np.ndarray,
                                                            out_file_name: str, n_alleles_per_variant: np.ndarray,
                                                            header: str = "",
                                                            add_genotype_likelihoods: Optional[np.ndarray] = None,
                                                            ):
    assert len(n_alleles_per_variant) == len(variants)
    print("N numeric genotypes", len(numeric_genotypes))
    if not np.all(n_alleles_per_variant == 2):
        if add_genotype_likelihoods is not None:
            logging.warning("Genotype likelihoods are not supported for multiallelic variants. Input vcf should be biallelic. Will not write genotype likelihoods")
            add_genotype_likelihoods = None

    string_genotypes = convert_biallelic_numeric_genotypes_to_multialellic_string_genotypes(n_alleles_per_variant, numeric_genotypes)

    print("N string genotypes: ", len(string_genotypes))
    print(string_genotypes)
    print("N variants:", len(variants))
    write_vcf(variants, string_genotypes, out_file_name, header, add_genotype_likelihoods=add_genotype_likelihoods)


def write_vcf(variants: SimpleVcfEntry, string_genotypes: bnp.EncodedRaggedArray,
              out_file_name: str, header: str = "",
              add_genotype_likelihoods: Optional[np.ndarray] = None):
    """Numeric genotypes: 1: 0/0, 2: 1/1, 3: 0/1 """



    #string_genotypes = ["0/0", "1/1", "0/1"]
    genotypes = string_genotypes  # [string_genotypes[g-1] for g in numeric_genotypes]
    format = ["GT" for i in range(len(variants))]

    if add_genotype_likelihoods is not None:
        logging.info("Writing genotype likelyhoods to file")
        p = add_genotype_likelihoods
        logging.info(p)
        genotype_likelihoods = p * np.log10(np.e)
        logging.info(genotype_likelihoods[genotype_likelihoods < -60])
        genotype_likelihoods[genotype_likelihoods < -60] = -60
        logging.info(genotype_likelihoods[genotype_likelihoods < -60])
        genotype_likelihoods[genotype_likelihoods == 0] = -0.0001
        logging.info(genotype_likelihoods)
        gl_strings = (",".join(str(p) if p != 0 else "-0.01" for p in genotype_likelihoods[i]) for i in range(len(variants)))
        genotypes = [f"{genotype}:{gl}" for genotype, gl in zip(genotypes, gl_strings)]
        format = ["GT:GL" for i in range(len(variants))]

    entry = VcfEntryWithSingleIndividualGenotypes(
        variants.chromosome,
        variants.position,
        ["variant" + str(i) for i in range(len(variants))],
        variants.ref_seq,
        variants.alt_seq,
        ["." for i in range(len(variants))],
        ["." for i in range(len(variants))],
        ["." for i in range(len(variants))],
        format,
        genotypes
    )

    entry.set_context("header", header)
    with bnp.open(out_file_name, "w", VcfWithSingleIndividualBuffer) as f:
        f.write(entry)


def create_vcf_header_with_sample_name(existing_vcf_header, sample_name, add_genotype_likelyhoods=False):
    header_lines = existing_vcf_header.split("\n")

    if add_genotype_likelyhoods:
        header_lines = header_lines[:-2] + vcf_pl_and_gl_header_lines() + header_lines[-2:]

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




