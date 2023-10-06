import logging
from dataclasses import dataclass

from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from kage.preprocessing.variants import Variants
import tqdm

logging.basicConfig(level=logging.INFO)
from typing import Dict, List, Tuple

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
    def from_multiallelic_vcf(cls, file_name, convert_to_biallelic=True):
        """
        Also works for nonmultiallelic. Will split multiallelic entries into biallelic if there are any.
        """
        logging.info("Redding vcf")
        vcf_entry = bnp.open(file_name, buffer_type=bnp.io.VCFBuffer).read()
        
        # read genotypes using VcfEntryWithSingleIndividualGenotypes
        # encode genotypes manually since bionumpy does not support many alleles
        # encode missing haplotype
        logging.info("Reading genotypes")
        vcf_with_genotypes = bnp.open(file_name, buffer_type=VcfWithSingleIndividualBuffer).read()
        logging.info("Processing genotypes")
        genotypes = vcf_with_genotypes.genotype.tolist()
        genotypes = (normalize_genotype(g) for g in genotypes)
        genotypes = normalized_genotypes_to_haplotype_matrix(genotypes)
        logging.info("Making haplotype matrix")
        haplotype_matrix = SparseHaplotypeMatrix.from_nonsparse_matrix(genotypes)
        if convert_to_biallelic:
            return cls.from_multiallelic_variants_and_haplotype_matrix(vcf_entry, haplotype_matrix)
        else:
            return cls.from_multiallelic_variants_and_haplotype_matrix_without_biallelic_converion(vcf_entry, haplotype_matrix)

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
    def from_multiallelic_variants_and_haplotype_matrix_without_biallelic_converion(cls, variants: VcfEntry, haplotype_matrix: SparseHaplotypeMatrix, missing_genotype_encoding=127):
        """
        Normalizes multiallelic variants by sorting on alternative allele. Converts genotypes according to sorting
        so that the same multiallelic variants with different sorting can be compared.
        """
        index = {}
        n_changed_order = 0
        for variant, haplotypes in zip(variants, haplotype_matrix.to_matrix()):
            alt_seqs = variant.alt_seq.to_string().split(",")
            sorting = np.argsort(alt_seqs)
            if not np.all(sorting == np.arange(len(sorting))):
                n_changed_order += 1
            new_alt_seqs = [alt_seqs[i] for i in sorting]
            mapping = {sorting[i]+1: i+1 for i in range(len(sorting))}
            # recode genotype to new sorting. 0 should be 0 and missing should be missing
            alleles = sorted([mapping[haplotype] if haplotype != 0 and haplotype != missing_genotype_encoding else haplotype for haplotype in haplotypes])
            key = f"{variant.chromosome.to_string()}-{variant.position}-{variant.ref_seq.to_string()}-{'-'.join(new_alt_seqs)}"

            allele1 = str(alleles[0])
            if allele1 == str(missing_genotype_encoding):
                allele1 = "."
            allele2 = str(alleles[1])
            if allele2 == str(missing_genotype_encoding):
                allele2 = "."
            index[key] = f"{allele1}/{allele2}"

        logging.info("Normalized %d multiallelic variants by changing order of alt alleles" % n_changed_order)

        return cls(index)

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

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    def _preprocess(self):
        n_with_missing = 0
        n_not_found_in_sample = 0
        self._confusion_matrix = {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "hetero": 0,
            "hetero_correct": 0,
            "homo_ref": 0,
            "homo_ref_correct": 0,
            "homo_alt": 0,
            "homo_alt_correct": 0,
            "missing": 0,
            "missing_correct": 0
        }

        truth = self._truth
        for i, (key, t) in tqdm.tqdm(enumerate(truth.items()), desc="Running comparison", total=len(truth.items())):
            if isinstance(t, BiallelicVariant):
                t = t.genotype_string()

            if key not in self._sample:
                n_not_found_in_sample += 1
                # did not genotype, treat as 0/0
                g = "0/0"
            else:
                g = self._sample[key]
                if isinstance(g, BiallelicVariant):
                    g = g.genotype_string()

            if "." in t:
                n_with_missing += 1
                #continue
                self._confusion_matrix["missing"] += 1
                if "." in g:
                    # if any allele is missing, this is "correct" (same way PanGenie defines correct missing)
                    self._confusion_matrix["missing_correct"] += 1
            elif t == "0/0":
                self._confusion_matrix["homo_ref"] += 1
                if g == "0/0":
                    self._confusion_matrix["homo_ref_correct"] += 1
            elif t.startswith("0/"):
                self._confusion_matrix["hetero"] += 1
                if g == t:
                    self._confusion_matrix["hetero_correct"] += 1
            else:
                self._confusion_matrix["homo_alt"] += 1
                if g == t:
                    self._confusion_matrix["homo_alt_correct"] += 1

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
        logging.info(f"{n_not_found_in_sample} truth genotypes not found in sample. These were set to 0/0")
        logging.info("Confusion matrix: %s" % self._confusion_matrix)

    def _validate(self):
        """
        Look for problems that can make the comparison invalid
        If the smaple has genotypes that the truth doesn't have, something is wrong
        """
        logging.info("Validating...")
        for key, g in tqdm.tqdm(self._sample.items()):
            if key not in self._truth:
                logging.error(f"Sample has genotype {g} for variant {key[0:100]}... not in truth")
                # try to find closest match for debugging
                s = "-".join(key.split("-")[0:3])
                matches = [k for k in self._truth._index if k.startswith(s)]
                for match in matches:
                    logging.error("   Match: %s" % match[0:100])

                raise Exception("Sample has genotype not in truth. Something is probably wrong")
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
    def concordance_hetero(self):
        return self._confusion_matrix["hetero_correct"] / self._confusion_matrix["hetero"] if self._confusion_matrix["hetero"] > 0 else 0

    @property
    def concordance_homo_ref(self):
        return self._confusion_matrix["homo_ref_correct"] / self._confusion_matrix["homo_ref"] if self._confusion_matrix["homo_ref"] > 0 else 0

    @property
    def concordance_homo_alt(self):
        return self._confusion_matrix["homo_alt_correct"] / self._confusion_matrix["homo_alt"] if self._confusion_matrix["homo_alt"] > 0 else 0

    @property
    def weighted_concordance(self):
        return (self.concordance_hetero + self.concordance_homo_alt + self.concordance_homo_ref) / 3

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


@dataclass
class BiallelicVariant:
    chromosome: str
    position: int
    reference_sequence: str
    alt_sequences: str
    genotype: List[int]

    def key(self):
        return f"{self.chromosome}-{self.position}-{self.reference_sequence}-{self.alt_sequences}"

    def genotype_string(self, missing_genotype_encoding=127):
        alleles = self.genotype
        allele1 = str(alleles[0])
        if allele1 == str(missing_genotype_encoding):
            allele1 = "."
        allele2 = str(alleles[1])
        if allele2 == str(missing_genotype_encoding):
            allele2 = "."
        return f"{allele1}/{allele2}"


@dataclass
class MultiAllelicVariant(BiallelicVariant):
    """
    Represents a multiallelic variant that can be uniquely identified
    by chromosome, position and reference sequence (which works when multiallelic variants are not overlapping)
    """
    alt_sequences: List[str]

    def key(self):
        return f"{self.chromosome}-{self.position}-{self.reference_sequence}"

    def normalize_using_reference_variant(self, reference_variant, missing_allele_encoding=127):
        """
        "Normalizes" using another  reference variant that is identical but that
        may have different order of alternative sequences. Uses the reference
        to reorder the alternative sequences and recode the genotype accordingly.
        """
        assert self.key() == reference_variant.key()
        allele_mapping = {seq: i for i, seq in enumerate(reference_variant.alt_sequences)}
        for seq in self.alt_sequences:
            assert seq in allele_mapping, f"Allele {seq} not in reference variant {reference_variant} but is in {self}"

        new_alt_sequences = reference_variant.alt_sequences  # [self.alt_sequences[allele_mapping[seq]] for seq in self.alt_sequences]
        new_alleles = sorted([allele_mapping[self.alt_sequences[allele-1]]+1 if allele != 0 and allele != missing_allele_encoding
                       else allele for allele in self.genotype])

        self.alt_sequences = new_alt_sequences
        self.genotype = new_alleles


class IndexedGenotypes2(IndexedGenotypes):
    """Similar to IndexGenotypes but represents variants using MultiAllelicVariant object.
    Makes it easier to normalize variants that have different order of alternative alleles.
    """
    def __init__(self, index: Dict[str, MultiAllelicVariant]):
        self._index = index

    @classmethod
    def from_multiallelic_variants_and_haplotype_matrix_without_biallelic_converion(cls, variants: VcfEntry,
                                                                                    haplotype_matrix: SparseHaplotypeMatrix,
                                                                                    missing_genotype_encoding=127):
        """
        Normalizes multiallelic variants by sorting on alternative allele. Converts genotypes according to sorting
        so that the same multiallelic variants with different sorting can be compared.
        """
        index = {}
        n_changed_order = 0
        for variant, haplotypes in zip(variants, haplotype_matrix.to_matrix()):
            v = MultiAllelicVariant(variant.chromosome.to_string(), variant.position, variant.ref_seq.to_string(),
                                    variant.alt_seq.to_string().split(","), list(haplotypes))
            index[v.key()] = v

        return cls(index)


    def normalize_against_reference_variants(self, reference_variants: 'IndexedGenotypes2'):
        for key, variant in self.items():
            assert key in reference_variants, f"Variant {key} not in reference variants"
            reference_variant = reference_variants[key]
            variant.normalize_using_reference_variant(reference_variant)

    @classmethod
    def from_biallelic_vcf(cls, file_name):
        vcf_with_genotypes = bnp.open(file_name, buffer_type=VcfWithSingleIndividualBuffer).read()
        logging.info("Processing genotypes")
        genotypes = vcf_with_genotypes.genotype.tolist()
        genotypes = (normalize_genotype(g) for g in genotypes)
        logging.info("Making haplotype matrix")
        genotypes = normalized_genotypes_to_haplotype_matrix(genotypes)
        index = {}

        for variant, haplotypes in tqdm.tqdm(zip(vcf_with_genotypes, genotypes), total=len(genotypes)):
            assert "," not in variant.alt_seq.to_string()
            v = BiallelicVariant(variant.chromosome.to_string(), variant.position, variant.ref_seq.to_string(),
                                 variant.alt_seq.to_string(), list(haplotypes))
            index[v.key()] = v

        return cls(index)

