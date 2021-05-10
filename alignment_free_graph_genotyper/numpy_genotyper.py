import logging
from .genotyper import Genotyper
from scipy.stats import binom
import numpy as np

class NumpyGenotyper(Genotyper):

    def compute_marginal_probabilities(self):
        # find expected count ref, alt for the three different genotypes
        ref_nodes = self._variant_to_nodes.ref_nodes
        alt_nodes = self._variant_to_nodes.var_nodes

        model = self._node_count_model
        expected_count_on_variant_ref_nodes = {}
        expected_count_on_variant_alt_nodes = {}

        expected_count_on_variant_ref_nodes["homo_ref"] = model.counts_homo_ref[ref_nodes] + 0.1
        expected_count_on_variant_ref_nodes["homo_alt"] = model.counts_homo_alt[ref_nodes] + 0.01
        expected_count_on_variant_ref_nodes["hetero"] = model.counts_hetero[ref_nodes] + 0.055

        expected_count_on_variant_alt_nodes["homo_ref"] = model.counts_homo_ref[alt_nodes] + 0.01
        expected_count_on_variant_alt_nodes["homo_alt"] = model.counts_homo_alt[alt_nodes] + 0.1
        expected_count_on_variant_alt_nodes["hetero"] = model.counts_hetero[alt_nodes] + 0.055

        observed_ref_nodes = self._node_counts.get_node_count_array()[ref_nodes]
        observed_alt_nodes = self._node_counts.get_node_count_array()[alt_nodes]

        # binomial
        n = observed_alt_nodes + observed_ref_nodes
        k = observed_ref_nodes
        marginal_probs = np.zeros((3, len(ref_nodes)))  # marginal probs go into matrix, each row is one genotype, columns are variants
        for i, genotype in enumerate(["homo_ref", "homo_alt", "hetero"]):
            logging.info("Computing marginal probs for genotypes %s" % genotype)
            p = expected_count_on_variant_ref_nodes[genotype] / (expected_count_on_variant_ref_nodes[genotype] + expected_count_on_variant_alt_nodes[genotype])
            marginal_probs[i] = binom.pmf(k, n, p)

        self.marginal_probs = marginal_probs

    def _genotype_biallelic_variant(self, variant_id, a_priori_homozygous_ref, a_priori_homozygous_alt,
                                a_priori_heterozygous):

        p_counts_given_homozygous_ref = self.marginal_probs[0, variant_id]
        p_counts_given_homozygous_alt = self.marginal_probs[1, variant_id]
        p_counts_given_heterozygous = self.marginal_probs[2, variant_id]

        prob_posteriori_heterozygous = a_priori_heterozygous * p_counts_given_heterozygous
        prob_posteriori_homozygous_alt = a_priori_homozygous_alt * p_counts_given_homozygous_alt
        prob_posteriori_homozygous_ref = a_priori_homozygous_ref * p_counts_given_homozygous_ref

        sum_of_posteriori = prob_posteriori_homozygous_ref + prob_posteriori_heterozygous + prob_posteriori_homozygous_alt

        prob_posteriori_heterozygous /= sum_of_posteriori
        prob_posteriori_homozygous_alt /= sum_of_posteriori
        prob_posteriori_homozygous_ref /= sum_of_posteriori

        sum_of_probs = prob_posteriori_homozygous_ref + prob_posteriori_homozygous_alt + prob_posteriori_heterozygous

        if abs(sum_of_probs - 1.0) > 0.01:
            logging.warning("Probs do not sum to 1.0: Sum is %.5f" % sum_of_probs)

        if prob_posteriori_homozygous_ref > prob_posteriori_homozygous_alt and prob_posteriori_homozygous_ref > prob_posteriori_heterozygous:
            predicted_genotype = "0/0"
        elif prob_posteriori_homozygous_alt > prob_posteriori_heterozygous and prob_posteriori_homozygous_alt > 0.0:
            predicted_genotype = "1/1"
        elif prob_posteriori_heterozygous > 0.0:
            predicted_genotype = "0/1"
        else:
            # logging.warning("All probs are zero for variant at node %d/%d." % (reference_node, variant_node))
            # logging.warning("Model counts ref: %d/%d/%d." % (self._node_count_model.counts_homo_ref[reference_node], self._node_count_model.counts_homo_alt[reference_node], self._node_count_model.counts_hetero[reference_node]))
            # logging.warning("Model counts var: %d/%d/%d." % (self._node_count_model.counts_homo_ref[variant_node], self._node_count_model.counts_homo_alt[variant_node], self._node_count_model.counts_hetero[variant_node]))
            # logging.warning("Node counts: %d/%d." % (self._node_counts.node_counts[reference_node], self._node_counts.node_counts[variant_node]))
            # logging.warning("Posteriori probs: %.4f, %.4f, %.4f" % (p_counts_given_homozygous_ref, p_counts_given_homozygous_alt, p_counts_given_heterozygous))
            # logging.warning("A priori probs  : %.4f, %.4f, %.4f" % (a_priori_homozygous_ref, a_priori_homozygous_alt, a_priori_heterozygous))
            # logging.warning("%.5f / %.5f / %.5f" % (
            # prob_posteriori_homozygous_ref, prob_posteriori_homozygous_alt, prob_posteriori_heterozygous))
            predicted_genotype = "0/0"

        return predicted_genotype


    def genotype(self):
        self.compute_marginal_probabilities()

        variant_id = -1
        for i, variant in enumerate(self._variants):
            if i % 100000 == 0 and i > 0:
                logging.info("%d variants genotyped" % i)

            variant_id += 1
            variant_id = variant.vcf_line_number
            self._genotypes_called_at_variant.append(0)
            assert "," not in variant.variant_sequence, "Only biallelic variants are allowed. Line is not bialleleic"

            prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_most_similar_previous_variant(variant_id)

            predicted_genotype = self._genotype_biallelic_variant(variant_id, prob_homo_ref, prob_homo_alt,
                                                                  prob_hetero)

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

            if self._most_similar_variant_lookup is not None:
                self._genotypes_called_at_variant[variant_id] = numeric_genotype
