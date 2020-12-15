from .count_genotyper import get_variant_type
import logging
from collections import defaultdict
from graph_kmer_index import kmer_hash_to_sequence, sequence_to_kmer_hash


class TruthRegions:
    def __init__(self, file_name):
        self.regions = []
        f = open(file_name)

        for line in f:
            l = line.split()
            start = int(l[1])
            end = int(l[2])

            self.regions.append((start, end))

    def is_inside_regions(self, position):
        for region in self.regions:
            if position >= region[0] and position < region[1]:
                return True
        return False

class VariantGenotype:
    def __init__(self, position, ref_sequence, variant_sequence, genotype, type=""):
        self.position = position
        self.ref_sequence = ref_sequence
        self.variant_sequence = variant_sequence
        self.genotype = genotype
        if self.genotype == "1|0":
            self.genotype = "0|1"

        self.type = type

    def id(self):
        return (self.position, self.ref_sequence, self.variant_sequence)


    def __str__(self):
        return "%d %s/%s %s %s" % (self.position, self.ref_sequence, self.variant_sequence, self.genotype, self.type)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if self.id() == other.id():
            if self.genotype == other.genotype:
                return True
        else:
            logging.error("ID mismatch: %s != %s" % (self.id(), other.id()))

        return False


class GenotypeCalls:
    def __init__(self, variant_genotypes):
        self.variant_genotypes = variant_genotypes
        self._index = {}
        self.make_index()

    def make_index(self):
        logging.info("Making vcf index")
        for variant in self.variant_genotypes:
            self._index[variant.id()] = variant
        logging.info("Done making vcf index")

    def has_variant(self, variant_genotype):
        if variant_genotype.id() in self._index:
            return True

        return False

    def has_variant_genotype(self, variant_genotype):
        if self.has_variant(variant_genotype) and self._index[variant_genotype.id()].genotyep == variant_genotype.genotype:
            return True

        return False

    def get(self, variant):
        return self._index[variant.id()]

    def __iter__(self):
        return self.variant_genotypes.__iter__()

    def __next__(self):
        return self.variant_genotypes.__next__()

    @classmethod
    def from_vcf(cls, vcf_file_name):
        variant_genotypes = []

        f = open(vcf_file_name)
        for line in f:
            if line.startswith("#"):
                continue

            l = line.split()
            position = int(l[1])
            ref_sequence = l[3].lower()
            variant_sequence = l[4].lower()

            if len(l) >= 10:
                genotype = l[9].split(":")[0].replace("/", "|")
            else:
                genotype = ""

            variant_genotypes.append(VariantGenotype(position, ref_sequence, variant_sequence, genotype, get_variant_type(line)))

        return cls(variant_genotypes)


class KmerAnalyser:
    def __init__(self, graph, k, variants, kmer_index, reverse_kmer_index, predicted_genotypes, truth_genotypes, truth_regions):
        self.graph = graph
        self.k = k
        self.variants = variants
        self.kmer_index = kmer_index
        self.reverse_kmer_index = reverse_kmer_index
        self.n_ambiguous = 0
        self.n_ambiguous_wrongly_genotyped = 0
        self.n_deletions = 0
        self.predicted_genotypes = predicted_genotypes
        self.truth_regions = truth_regions
        self.truth_genotypes = truth_genotypes
        self.n_truth_variants = defaultdict(int)
        self.n_predicted_variants = defaultdict(int)
        self.n_correct_genotypes = defaultdict(int)
        self.false_positives = defaultdict(int)

    def print_report(self):
        for type in ["SNP", "DELETION"]:
            print("Type: %s" % type)
            print("N truth variants: %d" % self.n_truth_variants[type])
            print("N predicted variants: %d" % self.n_predicted_variants[type])
            print("N correctly predicted: %d" % self.n_correct_genotypes[type])
            print("N false postives: %d" % self.false_positives[type])

    def analyse_variant(self, reference_node, variant_node, variant):
        reference_kmers = set(self.reverse_kmer_index.get_node_kmers(reference_node))
        variant_kmers = set(self.reverse_kmer_index.get_node_kmers(variant_node))

        if self.truth_genotypes.has_variant(variant):
            self.n_truth_variants[variant.type] += 1
        if self.predicted_genotypes.has_variant(variant):
            self.n_predicted_variants[variant.type] += 1

            if self.truth_genotypes.has_variant(variant):
                if self.truth_genotypes.get(variant) == self.predicted_genotypes.get(variant):
                    self.n_correct_genotypes[variant.type] += 1
                #else:
                #logging.warning("Wrong genotype: %s / %s" % (self.truth_genotypes.get(variant), self.predicted_genotypes.get(variant)))

        if self.predicted_genotypes.has_variant(variant) and not self.truth_genotypes.has_variant(variant):
            if self.truth_regions.is_inside_regions(variant.position) and self.predicted_genotypes.get(variant).genotype != "0|0":

                if False and variant.type == "SNP":
                    logging.warning("False positive genotype!")
                    logging.warning("Predicted: %s" %  (self.predicted_genotypes.get(variant)))
                    logging.warning("Kmers on ref/variant node %d/%d: %s / %s" % (reference_node, variant_node, reference_kmers, variant_kmers))
                    logging.warning("Kmer sequences on variant node: %s" % [(hash, kmer_hash_to_sequence(hash, 31)) for hash in variant_kmers])
                    logging.warning("Kmer sequences on ref     node: %s" % [kmer_hash_to_sequence(hash, 31) for hash in reference_kmers])
                    if variant.type == "SNP":
                        prev_node_difference = 2
                    elif variant.type == "DELETION":
                        prev_node_difference = 1
                    prev_node = self.graph.get_node_at_ref_offset(variant.position - prev_node_difference)
                    next_node = self.graph.get_node_at_ref_offset(variant.position + self.graph.get_node_size(reference_node) + prev_node_difference)
                    logging.warning("Ref nodes in area: %s" % str([self.graph.get_node_sequence(node) for node in [prev_node, reference_node, next_node]]))

                self.false_positives[variant.type] += 1


        if len(reference_kmers.intersection(variant_kmers)) > 0:
            logging.warning("----------")
            logging.warning("Variant and reference node share kmers on %d/%d: %s / %s" % (reference_node, variant_node, reference_kmers, variant_kmers))
            # Check if they have any of the same kmers
            if len(reference_kmers) == 0:
                logging.warning("Variant %s %d/%d has 0 reference kmers" % (variant, reference_node, variant_node))
            if len(reference_kmers) == 0:
                logging.warning("Variant %s %d/%d has 0 variant kmers" % (variant, reference_node, variant_node))

            if variant.type == "DELETION":
                logging.warning("Reference node sequence: %s" % self.graph.get_node_sequence(reference_node))
                logging.warning("Next reference node sequence: %s" % self.graph.get_node_sequence(self.graph.get_edges(reference_node)[0]))

                self.n_ambiguous += 1

                if self.predicted_genotypes.has_variant(variant):
                    logging.warning("Predicted: %s" % self.predicted_genotypes.get(variant))
                    if (not self.truth_genotypes.has_variant(variant) and self.predicted_genotypes.get(variant).genotype != "0|0") \
                            or (self.truth_genotypes.has_variant(variant) and self.predicted_genotypes.get(variant) != self.truth_genotypes.get(variant)):
                        self.n_ambiguous_wrongly_genotyped += 1

                if self.truth_genotypes.has_variant(variant):
                    logging.warning("Truth: %s" % self.predicted_genotypes.get(variant))

        #logging.info("Variant and reference node on %d/%d: %s / %s" % (reference_node, variant_node, reference_kmers, variant_kmers))

    def analyse_unique_kmers_on_variants(self):
        #for i, line in enumerate(open(self.vcf_file_name)):
        for i, variant in enumerate(self.variants.variant_genotypes):
            if i % 10000 == 0:
                logging.info("%d lines processed" % i)

            ref_allele = variant.ref_sequence
            variant_allele = variant.variant_sequence
            ref_offset = variant.position - 1
            assert "," not in variant_allele, "Only biallelic variants are allowed. Line is not bialleleic"

            if variant.type == "SNP":
                reference_node, variant_node = self.graph.get_snp_nodes(ref_offset, variant_allele)
            elif variant.type == "DELETION":
                self.n_deletions += 1
                reference_node, variant_node = self.graph.get_deletion_nodes(ref_offset, len(ref_allele)-1)

            else:
                continue

            self.analyse_variant(reference_node, variant_node, variant)

        logging.info("N deletions: %d" % self.n_deletions)
        logging.info("N ambiguous deletions: %d" % self.n_ambiguous)
        logging.info("N ambiguous deletions wrongly genotyped: %d" % self.n_ambiguous_wrongly_genotyped)
        print(self.n_truth_variants)
        self.print_report()
