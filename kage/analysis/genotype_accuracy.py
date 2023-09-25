import logging

from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.preprocessing.variants import Variants

logging.basicConfig(level=logging.INFO)
from typing import Dict


import bionumpy as bnp
import numpy as np
from ..io import VcfEntryWithSingleIndividualGenotypes as VcfEntry
from ..io import VcfWithSingleIndividualBuffer


def read_vcf_with_genotypes(file_name) -> VcfEntry:
    return bnp.open(file_name, buffer_type=VcfWithSingleIndividualBuffer).read()


def normalize_genotype(genotype):
    if ":" in genotype:
        genotype = genotype.split(":")[0]

    if genotype == ".":
        return "./."
    else:
        # sort by smallest allele first to make comparison of
        # unphased genontypes possible
        genotype = genotype.replace("|", "/")
        alleles = genotype.split("/")
        alleles = sorted(alleles)
        return "/".join(alleles)


def normalized_genotypes_to_haplotype_matrix(genotypes, encode_missing_as=127):
    out = []
    for genotype in genotypes:
        assert "/" in genotype
        allele0 = genotype.split("/")[0]
        allele1 = genotype.split("/")[1]
        if allele0 == ".":
            allele0 = encode_missing_as
        if allele1 == ".":
            allele1 = encode_missing_as

        numeric_genotype = [int(allele0), int(allele1)]
        out.append(numeric_genotype)

    return np.array(out)


class IndexedGenotypes:
    """Enables lookup from an index to a genotype.
    Genotypes are normalized for comparison"""
    def __init__(self, index: Dict[str, str]):
        self._index = index

    @classmethod
    def from_vcf_entry(cls, vcf_entry: VcfEntry):
        """
        Only for simple biallelic vcfs that can be compared directly
        """
        # hash chromosome, start, ref, alt
        index = {}
        for variant in vcf_entry:
            assert "," not in variant.alt_seq.to_string(), "Only biallelic variants supported. Use from_multiallelic_vcf"
            lookup_string = f"{variant.chromosome.to_string()}-{variant.position}-{variant.ref_seq.to_string()}-{variant.alt_seq.to_string()}"
            index[lookup_string] = normalize_genotype(variant.genotype.to_string())

        return cls(index)

    @classmethod
    def from_multiallelic_vcf(cls, file_name):
        """
        Also works for nonmultiallelic. Will split multiallelic entries into biallelic if there are any.
        """
        vcf_entry = bnp.open(file_name, buffer_type=bnp.io.delimited_buffers.VCFBuffer).read()
        
        # read genotypes using VcfEntryWithSingleIndividualGenotypes
        # encode genotypes manually since bionumpy does not support many alleles
        # encode missing haplotype
        vcf_with_genotypes = bnp.open(file_name, buffer_type=VcfWithSingleIndividualBuffer).read()
        genotypes = vcf_with_genotypes.genotype.tolist()
        genotypes = (normalize_genotype(g) for g in genotypes)
        genotypes = normalized_genotypes_to_haplotype_matrix(genotypes)
        haplotype_matrix = SparseHaplotypeMatrix.from_nonsparse_matrix(genotypes)
        #haplotype_matrix = SparseHaplotypeMatrix.from_vcf(file_name)
        return cls.from_multiallelic_variants_and_haplotype_matrix(vcf_entry, haplotype_matrix)

    @classmethod
    def from_vcf(cls, file_name):
        vcf_entry = read_vcf_with_genotypes(file_name)
        return cls.from_vcf_entry(vcf_entry)

    def __getitem__(self, key):
        return self._index[key]

    def __contains__(self, item):
        return item in self._index

    def items(self):
        return self._index.items()

    @classmethod
    def from_multiallelic_variants_and_haplotype_matrix(cls, variants: VcfEntry, haplotype_matrix: SparseHaplotypeMatrix, missing_genotype_encoding=127):
        """
        Creates biallelic variants and haplotype matrix.
        To be used when multiallelic variants are not normalized (e.g. different order of alt alleles)
        """
        biallelic_variants, n_alleles_per_variant = Variants.from_multiallelic_vcf_entry(variants, True)
        biallelic_alleles = haplotype_matrix.to_biallelic(n_alleles_per_variant+1).to_matrix()
        # sort by lowest allele first to normalize
        biallelic_alleles = np.sort(biallelic_alleles, axis=1)

        index = {}
        for variant, alleles in zip(biallelic_variants, biallelic_alleles):
            assert "," not in variant.alt_seq.to_string()
            key = f"{variant.chromosome.to_string()}-{variant.position}-{variant.ref_seq.to_string()}-{variant.alt_seq.to_string()}"
            allele1 = str(alleles[0])
            # missing is encoded as 10
            if allele1 == str(missing_genotype_encoding):
                allele1 = "."
            allele2 = str(alleles[1])
            if allele2 == str(missing_genotype_encoding):
                allele2 = "."
            index[key] = f"{allele1}/{allele2}"

        return cls(index)


class GenotypeAccuracy:
    def __init__(self, true_genotypes: IndexedGenotypes, inferred_genotypes: IndexedGenotypes):
        self._truth = true_genotypes
        self._sample = inferred_genotypes
        self._confusion_matrix = None
        self._out_report = {
            "false_negatives": [],
            "false_positives": []
        }
        self._preprocess()
        self._validate()

    def _preprocess(self):
        n_with_missing = 0
        self._confusion_matrix = {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0
        }

        truth = self._truth
        for i, (key, t) in enumerate(truth.items()):
            if key not in self._sample:
                # did not genotype, treat as 0/0
                g = "0/0"
            else:
                g = self._sample[key]

            if "." in t:
                n_with_missing += 1
                continue

            if t == g and t != "0/0":
                self._confusion_matrix["true_positive"] += 1
            elif t == '0/0' and g != '0/0':
                self._confusion_matrix["false_positive"] += 1
                self._out_report["false_positives"].append(i)
            elif t != '0/0' and g != t:
                self._confusion_matrix["false_negative"] += 1
                self._out_report["false_negatives"].append(i)
            elif t == "0/0" and g == "0/0":
                self._confusion_matrix["true_negative"] += 1
            else:
                assert False, (t, g)

        logging.info(f"{n_with_missing} truth genotypes contained missing allele(s)")
        logging.info("Confusion matrix: %s" % self._confusion_matrix)

    def _validate(self):
        """
        Look for problems that can make the comparison invalid
        If the smaple has genotypes that the truth doesn't have, something is wrong
        """
        for key, g in self._sample.items():
            if key not in self._truth:
                raise Exception(f"Sample has genotype {g} for variant {key} not in truth")

    @property
    def true_positive(self):
        return self._confusion_matrix["true_positive"]

    @property
    def true_negative(self):
        return self._confusion_matrix["true_negative"]

    @property
    def false_positive(self):
        return self._confusion_matrix["false_positive"]

    @property
    def false_negative(self):
        return self._confusion_matrix["false_negative"]

    def recall(self):
        total = self._confusion_matrix["true_positive"] + self._confusion_matrix["false_negative"]
        if total == 0:
            logging.warning("No true positives or false negatives. Setting recall to 0")
            return 0
        return self._confusion_matrix["true_positive"] / total

    def precision(self):
        total = self._confusion_matrix["true_positive"] + self._confusion_matrix["false_positive"]
        if total == 0:
            logging.warning("No true positives or false positives. Setting precision to 0")
            return 0
        return self._confusion_matrix["true_positive"] / total

    def f1(self):
        if self.precision() == 0 and self.recall() == 0:
            return 0
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())

    def get_debug_report(self):
        return self._out_report

