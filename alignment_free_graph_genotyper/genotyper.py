import logging
from scipy.special import comb
from .variants import VcfVariant, VcfVariants
from collections import defaultdict
import numpy as np
from scipy.stats import poisson, binom
from obgraph import VariantNotFoundException


def parse_vcf_genotype(genotype):
    return genotype.replace("|", "/").replace("1/0", "1/0")


class Genotyper:
    def __init__(self, node_count_model, variants: VcfVariants, variant_to_nodes, node_counts, genotype_frequencies, most_similar_variant_lookup, variant_window_size=500, ):
        self._variants = variants
        self._node_count_model = node_count_model

        if self._node_count_model is None:
            logging.warning("No node count model specified: Will use pseudocounts")

        self._genotype_frequencies = genotype_frequencies
        self._variant_to_nodes = variant_to_nodes
        self._node_counts = node_counts
        self.expected_read_error_rate = 0.03
        self._average_coverage = 7.0
        self._average_node_count_followed_node = 7.0
        self._variant_window_size = variant_window_size
        self._individuals_with_genotypes = []
        self._variant_counter = 0
        self._most_similar_variant_lookup = most_similar_variant_lookup
        self._genotypes_called_at_variant = []  # position is variant id, genotype is 1,2,3 (homo ref, homo alt, hetero)

    def get_node_count(self, node):
        return self._node_counts.node_counts[node]

    def expected_count_following_node(self, node):
        if self._node_count_model is None:
            return 100

        count = self._average_coverage * self._node_count_model.node_counts_following_node[node]
        count += 0.5
        return count

    def expected_count_not_following_node(self, node):
        if self._node_count_model is None:
            return 0.3

        count = self._average_coverage * self._node_count_model.node_counts_not_following_node[node]
        count += 0.025
        return count

    def prob_observing_counts(self, ref_node, variant_node, genotype, type="binomial"):
        ref_count = self.get_node_count(ref_node)
        variant_count = self.get_node_count(variant_node)

        if genotype == "1/1":
            expected_count_alt = self.expected_count_following_node(variant_node) * 2  # double because both haplotype follow node
            expected_count_ref = self.expected_count_not_following_node(ref_node) * 2
        elif genotype == "0/0":
            expected_count_alt = self.expected_count_not_following_node(variant_node) * 2
            expected_count_ref = self.expected_count_following_node(ref_node) * 2
        elif genotype == "0/1":
            expected_count_alt = self.expected_count_following_node(variant_node) + \
                                 self.expected_count_not_following_node(variant_node)
            expected_count_ref = self.expected_count_following_node(ref_node) + \
                                 self.expected_count_not_following_node(ref_node)
        else:
            raise Exception("Unsupported genotype %s" % genotype)

        if type == "binomial":
            p = expected_count_ref / (expected_count_alt + expected_count_ref)
            k = int(ref_count)
            n = ref_count + variant_count
            return binom.pmf(k, n, p)

        return poisson.pmf(int(ref_count), expected_count_ref) * poisson.pmf(int(variant_count), expected_count_alt)


    def _genotype_biallelic_variant(self, reference_node, variant_node, a_priori_homozygous_ref, a_priori_homozygous_alt,
                                a_priori_heterozygous, debug=False):

        p_counts_given_homozygous_ref = self.prob_observing_counts(reference_node, variant_node, "0/0")
        p_counts_given_homozygous_alt = self.prob_observing_counts(reference_node, variant_node, "1/1")
        p_counts_given_heterozygous = self.prob_observing_counts(reference_node, variant_node, "0/1")

        prob_posteriori_heterozygous = a_priori_heterozygous * p_counts_given_heterozygous
        prob_posteriori_homozygous_alt = a_priori_homozygous_alt * p_counts_given_homozygous_alt
        prob_posteriori_homozygous_ref = a_priori_homozygous_ref * p_counts_given_homozygous_ref

        # Minimum count for genotyping
        if self.get_node_count(variant_node) <= 0:
            return "0/0"

        if prob_posteriori_homozygous_ref > prob_posteriori_homozygous_alt and prob_posteriori_homozygous_ref > prob_posteriori_heterozygous:
            return "0/0"
        elif prob_posteriori_homozygous_alt > prob_posteriori_heterozygous and prob_posteriori_homozygous_alt > 0.0:
            return "1/1"
        elif prob_posteriori_heterozygous > 0.0:
            return "0/1"
        else:
            logging.info("%.5f / %.5f / %.5f" % (
                prob_posteriori_homozygous_ref, prob_posteriori_homozygous_alt, prob_posteriori_heterozygous))
            return "0/0"

    def get_allele_frequencies_from_most_similar_previous_variant(self, variant_id):
        most_similar = self._most_similar_variant_lookup.get_most_similar_variant(variant_id)
        most_similar_genotype = self._genotypes_called_at_variant[most_similar]
        prob_same_genotype = self._most_similar_variant_lookup.prob_of_having_the_same_genotype_as_most_similar(variant_id)
        prob_same_genotype = min(0.999, max(prob_same_genotype, 0.001)) * 0.92
        orig_prob_homo_ref, orig_prob_homo_alt, orig_prob_hetero = self._genotype_frequencies.get_frequencies_for_variant(variant_id)

        # First init probs for having genotypes and not having same genotypes as most similar
        prob_homo_ref = (1 - prob_same_genotype) * orig_prob_homo_ref
        prob_homo_alt = (1 - prob_same_genotype) * orig_prob_homo_alt
        prob_hetero = (1 - prob_same_genotype) * orig_prob_hetero

        if most_similar_genotype == 1:
            prob_homo_ref += prob_same_genotype
        elif most_similar_genotype == 2:
            prob_homo_alt += prob_same_genotype
        elif most_similar_genotype == 3:
            prob_hetero += prob_same_genotype

        return prob_homo_ref, prob_homo_alt, prob_hetero

    def genotype(self):
        variant_id = -1
        for i, variant in enumerate(self._variants):
            if i % 1000 == 0:
                logging.info("%d variants genotyped" % i)

            variant_id += 1
            self._genotypes_called_at_variant.append(0)
            assert "," not in variant.variant_sequence, "Only biallelic variants are allowed. Line is not bialleleic"

            debug = False
            try:
                reference_node = self._variant_to_nodes.ref_nodes[variant_id]
                variant_node = self._variant_to_nodes.var_nodes[variant_id]
            except VariantNotFoundException:
                continue

            # Compute from actual node counts instead (these are from traversing the graph)
            #prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_haplotype_counts(reference_node, variant_node)
            prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_most_similar_previous_variant(variant_id)

            predicted_genotype = self._genotype_biallelic_variant(reference_node, variant_node, prob_homo_ref, prob_homo_alt,
                                                              prob_hetero, debug)
            #self.add_individuals_with_genotype(predicted_genotype, reference_node, variant_node)

            #print("%s\t%s" % ("\t".join(l[0:9]), predicted_genotype))
            variant.set_genotype(predicted_genotype)

            numeric_genotype = 0
            if predicted_genotype == "0/0":
                numeric_genotype = 1
            elif predicted_genotype == "1/1":
                numeric_genotype = 2
            elif predicted_genotype == "0/1":
                numeric_genotype = 3

            self._genotypes_called_at_variant[variant_id] = numeric_genotype
