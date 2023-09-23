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
        return "0/0"
    else:
        # sort by smallest allele first to make comparison of
        # unphased genontypes possible
        genotype = genotype.replace("|", "/").replace("1/0", "0/1")
        alleles = genotype.split("/")
        alleles = sorted(alleles)
        return "/".join(alleles)


class IndexedGenotypes:
    """Enables lookup from an index to a genotype.
    Genotypes are normalized for comparison"""
    def __init__(self, index: Dict[str, str]):
        self._index = index

    @classmethod
    def from_vcf_entry(cls, vcf_entry: VcfEntry):
        # hash chromosome, start, ref, alt
        index = {}
        for variant in vcf_entry:
            lookup_string = f"{variant.chromosome.to_string()}-{variant.position}-{variant.ref_seq.to_string()}-{variant.alt_seq.to_string()}"
            index[lookup_string] = normalize_genotype(variant.genotype.to_string())

        return cls(index)

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

    def _preprocess(self):
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

            # t = t.to_string().replace("|", "/").replace("1/0", "0/1")
            # g = g.to_string().replace("|", "/").replace("1/0", "0/1")

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
        return self._confusion_matrix["true_positive"] / (self._confusion_matrix["true_positive"] + self._confusion_matrix["false_negative"])

    def precision(self):
        return self._confusion_matrix["true_positive"] / (self._confusion_matrix["true_positive"] + self._confusion_matrix["false_positive"])

    def f1(self):
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())

    def get_debug_report(self):
        return self._out_report

