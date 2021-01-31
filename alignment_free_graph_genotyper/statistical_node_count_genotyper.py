import logging
from scipy.special import comb
from .variants import VariantGenotype
from collections import defaultdict
import numpy as np
from scipy.stats import poisson, binom
from obgraph import VariantNotFoundException

def parse_vcf_genotype(genotype):
    return genotype.replace("|", "/").replace("1/0", "1/0")


class StatisticalNodeCountGenotyper:
    def __init__(self, node_count_model, vcf_file, variant_to_nodes, node_counts, genotype_frequencies, most_similar_variant_lookup, variant_window_size=500):
        self._vcf_file = vcf_file
        self._node_count_model = node_count_model

        if self._node_count_model is None:
            logging.warning("No node count model specified: Will use pseudocounts")

        self._genotype_frequencies = genotype_frequencies
        #self._haplotype_counts = haplotype_counts
        #self._graph = graph
        self._variant_to_nodes = variant_to_nodes
        self._node_counts = node_counts
        #self._n_haplotypes = haplotype_counts.nodes.shape[0]
        #logging.info("There are %d haplotypes")
        #assert self._n_haplotypes < 10000, "Too many haplotypes. Is something wrong?"
        self.expected_read_error_rate = 0.03
        self._average_coverage = 7.0
        self._average_node_count_followed_node = 7.0
        self._variant_window_size = variant_window_size
        self._individuals_with_genotypes = []
        #self._n_individuals = self._n_haplotypes // 2
        #self._individual_counts = np.zeros((variant_window_size, self._n_individuals))
        self._variant_counter = 0
        self._most_similar_variant_lookup = most_similar_variant_lookup
        self._genotypes_called_at_variant = []  # position is variant id, genotype is 1,2,3 (homo ref, homo alt, hetero)

    def get_most_common_individual_on_previous_variants(self):
        most_common = np.argmax(np.sum(self._individual_counts, 0))
        return most_common

    def get_allele_frequencies_from_haplotype_counts(self, ref_node, variant_node):
        raise Exception("Not supported anymore")

        n_haplotypes_total = self._haplotype_counts.nodes.shape[0]

        if n_haplotypes_total == 0:
            return 0

        #n_ref_node = len(np.where(self._haplotype_counts.nodes == ref_node)[0])
        #n_var_node = len(np.where(self._haplotype_counts.nodes == variant_node)[0])
        n_ref_node = self._haplotype_counts.n_haplotypes_on_node[ref_node]
        n_var_node = self._haplotype_counts.n_haplotypes_on_node[variant_node]
        prob_ref = max(0.001, n_ref_node / n_haplotypes_total)
        prob_var = max(0.001, n_var_node / n_haplotypes_total)

        prob_homo_ref = prob_ref ** 2
        prob_homo_var = prob_var ** 2
        prob_hetero = prob_ref * prob_var

        return prob_homo_ref, prob_homo_var, prob_hetero

    def _store_processed_variant(self, line, edge):
        pass

    def get_node_count(self, node):
        return self._node_counts.node_counts[node]

    def prob_count_on_node_given_not_follwing_node(self, node, count):
        # Get expected rate on node given not following that node
        expected_rate = self._average_coverage * self._node_count_model.node_counts_not_following_node[node] / 6
        expected_rate = max(self._average_node_count_followed_node * 0.01, expected_rate)
        #logging.info("Expected rate not following node %d: %.3f" % (node, expected_rate))
        return poisson.pmf(count, expected_rate)

    def prob_count_on_node_given_follwing_node(self, node, count):
        # Get expected rate on node given not following that node
        expected_rate = self._average_coverage * self._node_count_model.node_counts_following_node[node] / 6
        # Expected rate should not be lower than average coverage
        expected_rate = max(self._average_node_count_followed_node, expected_rate)
        #logging.info("Expected rate following node %d: %.3f" % (node, expected_rate))

        return poisson.pmf(count, expected_rate)

    def expected_count_following_node(self, node):
        if self._node_count_model is None:
            return 100

        count = self._average_coverage * self._node_count_model.node_counts_following_node[node] / 1
        count += 0.5
        return count

    def expected_count_not_following_node(self, node):
        if self._node_count_model is None:
            return 0.3

        count = self._average_coverage * self._node_count_model.node_counts_not_following_node[node] / 1
        count += 0.025
        return count

    def prob_counts_given_hetero(self, ref_node, variant_node, ref_count, variant_count, type="binomial"):
        expected_count_alt = self.expected_count_following_node(variant_node) + self.expected_count_not_following_node(variant_node)
        expected_count_ref = self.expected_count_following_node(ref_node) + self.expected_count_not_following_node(ref_node)

        if type == "binomial":
            p = expected_count_ref / (expected_count_alt + expected_count_ref)
            k = int(ref_count)
            n = ref_count + variant_count
            return binom.pmf(k, n, p)

        return poisson.pmf(int(ref_count), expected_count_ref) * poisson.pmf(int(variant_count), expected_count_alt)

    def prob_counts_given_homo_alt(self, ref_node, variant_node, ref_count, variant_count, type="binomial"):
        expected_count_alt = self.expected_count_following_node(variant_node)
        expected_count_alt *= 2  # double if both haplotypes follow this node
        expected_count_ref = self.expected_count_not_following_node(ref_node)
        expected_count_ref *= 2

        if type == "binomial":
            p = expected_count_ref / (expected_count_alt + expected_count_ref)
            k = int(ref_count)
            n = ref_count + variant_count
            return binom.pmf(k, n, p)

        return poisson.pmf(int(ref_count), expected_count_ref) * poisson.pmf(int(variant_count), expected_count_alt)

    def prob_counts_given_homo_ref(self, ref_node, variant_node, ref_count, variant_count, type="binomial"):
        expected_count_ref = self.expected_count_following_node(ref_node)
        expected_count_ref *= 2  # double if both haplotypes follow this node
        expected_count_alt = self.expected_count_not_following_node(variant_node)
        expected_count_alt *= 2

        if type == "binomial":
            p = expected_count_ref / (expected_count_alt + expected_count_ref)
            k = int(ref_count)
            n = ref_count + variant_count
            return binom.pmf(k, n, p)

        return poisson.pmf(int(ref_count), expected_count_ref) * poisson.pmf(int(variant_count), expected_count_alt)

    def _genotype_biallelic_variant(self, reference_node, variant_node, a_priori_homozygous_ref, a_priori_homozygous_alt,
                                a_priori_heterozygous, debug=False):
        # logging.info("Genotyping biallelic SNP with nodes %d/%d and allele frequency %.5f" % (reference_node, variant_node, allele_frequency))
        allele_counts = [int(self.get_node_count(reference_node)), int(self.get_node_count(variant_node))]

        """
        tot_counts = sum(allele_counts)
        e = self.expected_read_error_rate
        # Simple when we have bialleleic. Formula for multiallelic given in malva supplmentary
        p_counts_given_homozygous_alt = comb(tot_counts, allele_counts[1]) * (1 - e) ** allele_counts[1] * e ** (
                tot_counts - allele_counts[1])
        p_counts_given_homozygous_ref = comb(tot_counts, allele_counts[0]) * (1 - e) ** allele_counts[0] * e ** (
                tot_counts - allele_counts[0])
        # p_counts_given_heterozygous = 1 - p_counts_given_homozygous_alt - p_counts_given_homozygous_ref
        p_counts_given_heterozygous = 1 * comb(tot_counts, allele_counts[0]) * ((1 - e) / 2) ** allele_counts[0] * (
                (1 - e) / 2) ** allele_counts[1]
        """

        """
        p_counts_given_homozygous_ref = self.prob_count_on_node_given_not_follwing_node(variant_node, allele_counts[1]) * self.prob_count_on_node_given_follwing_node(reference_node, allele_counts[0])
        p_counts_given_homozygous_alt = self.prob_count_on_node_given_not_follwing_node(reference_node, allele_counts[0]) * self.prob_count_on_node_given_follwing_node(variant_node, allele_counts[1])
        p_counts_given_heterozygous = self.prob_count_on_node_given_follwing_node(reference_node, allele_counts[0]) * self.prob_count_on_node_given_follwing_node(variant_node, allele_counts[1])
        """

        p_counts_given_homozygous_ref = self.prob_counts_given_homo_ref(reference_node, variant_node, allele_counts[0], allele_counts[1])
        p_counts_given_homozygous_alt = self.prob_counts_given_homo_alt(reference_node, variant_node, allele_counts[0], allele_counts[1])
        p_counts_given_heterozygous = self.prob_counts_given_hetero(reference_node, variant_node, allele_counts[0], allele_counts[1])

        if reference_node == 50391085:
            logging.info("Node counts: %s" % allele_counts)
            logging.info("P homo ref: %.18f" % p_counts_given_homozygous_ref)
            logging.info("P homo alt: %.18f" % p_counts_given_homozygous_alt)
            logging.info("P hetero: %.18f" % p_counts_given_heterozygous)
            logging.info("A priori probs: %.4f, %.4f, %.4f" % (a_priori_homozygous_ref, a_priori_homozygous_alt, a_priori_heterozygous))
            debug = True

        # Denominator in bayes formula
        prob_counts = a_priori_homozygous_ref * p_counts_given_homozygous_ref + \
                      a_priori_homozygous_alt * p_counts_given_homozygous_alt + \
                      a_priori_heterozygous * p_counts_given_heterozygous

        prob_posteriori_heterozygous = a_priori_heterozygous * p_counts_given_heterozygous / prob_counts
        prob_posteriori_homozygous_alt = a_priori_homozygous_alt * p_counts_given_homozygous_alt / prob_counts
        prob_posteriori_homozygous_ref = a_priori_homozygous_ref * p_counts_given_homozygous_ref / prob_counts

        if debug:
            logging.info("==== Nodes: %d / %d" % (reference_node, variant_node))
            logging.info("Alle counts: %s" % allele_counts)
            logging.info("Prob counts given 00, 11, 01: %.5f, %.5f, %.10f" % (
                p_counts_given_homozygous_ref, p_counts_given_homozygous_alt, p_counts_given_heterozygous))
            logging.info("A priori probs: 00, 11, 01: %.3f, %.3f, %.3f" % (
                a_priori_homozygous_ref, a_priori_homozygous_alt, a_priori_heterozygous))
            logging.info("Prob of counts: %.3f" % prob_counts)
            logging.info("Posteriori probs for 00, 11, 01: %.4f, %.4f, %.4f" % (
                prob_posteriori_homozygous_ref, prob_posteriori_homozygous_alt, prob_posteriori_heterozygous))

        # Minimum counts for genotyping
        if allele_counts[1] <= 0:
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

        #logging.info("Most similar variant to %d is %d. Prob of having same genotype is: %.5f. Previous genotype was: %s" % (variant_id, most_similar, prob_same_genotype, most_similar_genotype))

        #orig_prob_homo_ref, orig_prob_homo_alt, orig_prob_hetero = self.get_allele_frequencies_from_haplotype_counts(reference_node, variant_node)
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
        for i, line in enumerate(open(self._vcf_file)):
            if i % 1000 == 0:
                logging.info("%d lines processed" % i)

            if line.startswith("#"):
                if line.startswith("#CHROM"):
                    self._n_haplotypes = (len(line.split()) - 9) * 2
                    logging.info("There are %d haplotypes in this file" % self._n_haplotypes)
                    print("\t".join(line.split()[0:9]).strip() + "\tDONOR")
                else:
                    print(line.strip())

                continue

            variant_id += 1
            self._genotypes_called_at_variant.append(0)

            variant = VariantGenotype.from_vcf_line(line)
            assert "," not in variant.variant_sequence, "Only biallelic variants are allowed. Line is not bialleleic"
            l = variant.vcf_line.split()


            debug = False
            try:
                #reference_node, variant_node = self._graph.get_variant_nodes(variant)
                reference_node = self._variant_to_nodes.ref_nodes[variant_id]
                variant_node = self._variant_to_nodes.var_nodes[variant_id]
            except VariantNotFoundException:
                continue

            # Compute from actual node counts instead (these are from traversing the graph)
            #prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_haplotype_counts(reference_node, variant_node)
            prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_most_similar_previous_variant(variant_id)

            """
            prob_homo_ref = self.compute_a_priori_probabilities("0/0", reference_node, variant_node)
            prob_homo_alt = self.compute_a_priori_probabilities("1/1", reference_node, variant_node)
            prob_hetero = self.compute_a_priori_probabilities("0/1", reference_node, variant_node)
            """

            predicted_genotype = self._genotype_biallelic_variant(reference_node, variant_node, prob_homo_ref, prob_homo_alt,
                                                              prob_hetero, debug)
            #self.add_individuals_with_genotype(predicted_genotype, reference_node, variant_node)

            print("%s\t%s" % ("\t".join(l[0:9]), predicted_genotype))

            numeric_genotype = 0
            if predicted_genotype == "0/0":
                numeric_genotype = 1
            elif predicted_genotype == "1/1":
                numeric_genotype = 2
            elif predicted_genotype == "0/1":
                numeric_genotype = 3

            self._genotypes_called_at_variant[variant_id] = numeric_genotype
