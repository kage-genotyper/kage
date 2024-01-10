import logging
import time
from collections import defaultdict
from typing import Optional

import bionumpy as bnp
import numpy as np
import scipy
from bionumpy.bnpdataclass import bnpdataclass
import npstructures as nps
from bionumpy.typing import SequenceID
from shared_memory_wrapper import to_file, from_file

from kage.util import vcf_pl_and_gl_header_lines
from bionumpy.bnpdataclass import BNPDataClass
import dataclasses


# old bnp vcfbuffer for now for subclassing SingleIndividual (new causes trouble with info)
class VCFBuffer(bnp.io.delimited_buffers.DelimitedBuffer):
    dataclass = bnp.datatypes.VCFEntry

    def get_data(self):
        data = super().get_data()
        data.position -= 1
        return data

    def get_field_by_number(self, field_nr: int, field_type: type=object):
        val = super().get_field_by_number(field_nr, field_type)
        if field_nr == 1:
            val -= 1
        return val

    @classmethod
    def from_data(cls, data: BNPDataClass) -> "DelimitedBuffer":
        data = dataclasses.replace(data, position=data.position+1)
        return super().from_data(data)



@bnpdataclass
class VcfEntryWithSingleIndividualGenotypes:
    """
    VCF entry with a single individual
    """
    chromosome: SequenceID
    position: int
    id: str
    ref_seq: str
    alt_seq: str
    quality: str
    filter: str
    info: str
    format: str
    genotype: str


class VcfWithSingleIndividualBuffer(VCFBuffer):
    dataclass = VcfEntryWithSingleIndividualGenotypes


@bnpdataclass
class VcfEntryWithInfo:
    chromosome: SequenceID
    position: int
    id: str
    ref_seq: str
    alt_seq: str
    quality: str
    filter: str
    info: str


class VcfWithInfoBuffer(VCFBuffer):
    dataclass = VcfEntryWithInfo


@bnpdataclass
class SimpleVcfEntry:
    chromosome: SequenceID
    position: int
    ref_seq: str
    alt_seq: str

    def to_file(self, file_name):
        return to_file((self.chromosome, self.position, self.ref_seq, self.alt_seq), file_name)

    @classmethod
    def from_file(cls, file_name):
        return cls(*from_file(file_name))

    @classmethod
    def from_vcf(cls, file_name: str):
        chunks = []
        for chunk in bnp.open(file_name):
            chunks.append(SimpleVcfEntry(chunk.chromosome, chunk.position, chunk.ref_seq, chunk.alt_seq))
        return np.concatenate(chunks)


def write_multiallelic_vcf_with_biallelic_numeric_genotypes(variants: SimpleVcfEntry, numeric_genotypes: np.ndarray,
                                                            out_file_name: str, n_alleles_per_variant: np.ndarray,
                                                            header: str = "",
                                                            add_genotype_likelihoods: Optional[np.ndarray] = None,
                                                            ignore_homo_ref=False
                                                            ):
    assert len(n_alleles_per_variant) == len(variants)
    if not np.all(n_alleles_per_variant == 2):
        if add_genotype_likelihoods is not None:
            logging.warning("Genotype likelihoods are not supported for multiallelic variants. Input vcf should be biallelic. Will not write genotype likelihoods")
            add_genotype_likelihoods = None

    string_genotypes = convert_biallelic_numeric_genotypes_to_multialellic_string_genotypes(n_alleles_per_variant, numeric_genotypes)
    write_vcf(variants, string_genotypes, out_file_name, header, add_genotype_likelihoods=add_genotype_likelihoods, ignore_homo_ref=ignore_homo_ref)


def write_vcf(variants: SimpleVcfEntry, string_genotypes: bnp.EncodedRaggedArray,
              out_file_name: str, header: str = "",
              add_genotype_likelihoods: Optional[np.ndarray] = None,
              ignore_homo_ref=False):
    """Numeric genotypes: 1: 0/0, 2: 1/1, 3: 0/1 """




    #string_genotypes = ["0/0", "1/1", "0/1"]
    genotypes = string_genotypes  # [string_genotypes[g-1] for g in numeric_genotypes]
    format = ["GT" for i in range(len(variants))]

    if add_genotype_likelihoods is not None:
        t0 = time.perf_counter()
        logging.info("Writing genotype likelyhoods to file")
        p = add_genotype_likelihoods
        # normalize genotype likelihoods so that they sum to 1
        p = p - scipy.special.logsumexp(p, axis=1)[:, np.newaxis]

        # probs are in loge, convert to minus log10 (?)
        genotype_likelihoods = p * np.log10(np.e)
        has_nan = np.where(np.any(np.isnan(genotype_likelihoods), axis=1))[0]
        if len(has_nan):
            for n in has_nan:
                logging.warning("Genotype likelyhood is nan")
                logging.info("%s %s" % (p[n], genotype_likelihoods[n]))
                genotype_likelihoods[n] = [-0.1, -0.1, -0.1]

        #genotype_likelihoods[genotype_likelihoods < -60] = -60
        #genotype_likelihoods[genotype_likelihoods == 0] = -0.0001
        gl_strings = (",".join(str(p) if p != 0 else "-0.00000000001" for p in genotype_likelihoods[i]) for i in range(len(variants)))
        format = ["GT:GL:GQ" for i in range(len(variants))]
        logging.info("Processing genotype likelihoods took %.3f sec" % (time.perf_counter()-t0))
        t0 = time.perf_counter()

        # genotype quality (probability that the call is incorrect)
        genotype_likelihoods_matrix = np.array(genotype_likelihoods)
        #genotype_qualities = (np.max(genotype_likelihoods_matrix, axis=1) - scipy.special.logsumexp(genotype_likelihoods_matrix, axis=1))
        sorted_gls = np.sort(genotype_likelihoods_matrix, axis=1)
        genotype_qualities = -scipy.special.logsumexp(sorted_gls[:, 0:2], axis=1).astype(int)  # sum of prob of two other genotypes
        logging.info("Computing GLs took %.3f sec" % (time.perf_counter()-t0))
        #genotype_qualities = -(np.max(genotype_likelihoods_matrix, axis=1) - scipy.special.logsumexp(genotype_likelihoods_matrix, axis=1))
        genotypes = [f"{genotype}:{gl}:{gq}" for genotype, gl, gq in zip(genotypes, gl_strings, map(str, genotype_qualities))]
        logging.info("Creating genotype strings took %.3f sec" % (time.perf_counter()-t0))

    entry = VcfEntryWithSingleIndividualGenotypes(
        variants.chromosome,
        variants.position,
        bnp.as_encoded_array(["variant" + str(i) for i in range(len(variants))]),
        variants.ref_seq,
        variants.alt_seq,
        bnp.as_encoded_array(["." for i in range(len(variants))]),
        bnp.as_encoded_array(["." for i in range(len(variants))]),
        bnp.as_encoded_array(["." for i in range(len(variants))]),
        bnp.as_encoded_array(format),
        bnp.as_encoded_array(genotypes)
    )

    if ignore_homo_ref:
        remove = bnp.str_equal(bnp.as_encoded_array(string_genotypes), "0/0")
        logging.info(f"Not writing {np.sum(remove)} genotypes that are 0/0")
        entry = entry[~remove]

    entry.set_context("header", header)
    with bnp.open(out_file_name, "w", VcfWithSingleIndividualBuffer) as f:
        f.write(entry)


def create_vcf_header_with_sample_name(existing_vcf_header, sample_name, add_genotype_likelyhoods=False) -> str:
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


@bnpdataclass
class CustomVCFEntry:
    chromosome: SequenceID
    position: int
    id: str
    ref_seq: str
    alt_seq: str



@bnpdataclass
class CustomVCFEntryWithInfo:
    chromosome: SequenceID
    position: int
    id: str
    ref_seq: str
    alt_seq: str
    quality: str
    filter: str
    info: str


class CustomVCFBuffer(bnp.io.delimited_buffers.DelimitedBuffer):
    """
    Custom Vcf buffer that skips lazy to save memory.
    To be used when only chromosome position and sequences are needed.
    """
    SKIP_LAZY = True
    dataclass = CustomVCFEntry

    def get_data(self):
        data = super().get_data()
        data.position -= 1
        return data

    def get_field_by_number(self, field_nr: int, field_type: type=object):
        val = super().get_field_by_number(field_nr, field_type)
        if field_nr == 1:
            val -= 1
        return val

    @classmethod
    def from_data(cls, data: BNPDataClass) -> "DelimitedBuffer":
        data = dataclasses.replace(data, position=data.position+1)
        return super().from_data(data)


class CustomVCFBufferWithInfo(CustomVCFBuffer):
    dataclass = CustomVCFEntryWithInfo


