import logging
from scipy.special import comb

from .node_count_model import GenotypeNodeCountModel
from obgraph.variants import VcfVariant, VcfVariants
from collections import defaultdict
import numpy as np
from scipy.stats import poisson, binom
from obgraph import VariantNotFoundException
from Bio.Seq import Seq
from .node_counts import NodeCounts


def parse_vcf_genotype(genotype):
    return genotype.replace("|", "/").replace("1/0", "1/0")


class Genotyper:
    def __init__(
        self,
        node_count_model,
        variants: VcfVariants,
        variant_to_nodes,
        node_counts,
        genotype_frequencies,
        most_similar_variant_lookup,
        variant_window_size=500,
    ):
        self._variants = variants
        self._node_count_model = node_count_model

        # if self._node_count_model is None:
        #    logging.warning("No node count model specified: Will use pseudocounts")

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
        self._genotypes_called_at_variant = (
            {}
        )  # index is variant id, genotype is 1,2,3 (homo ref, homo alt, hetero)
        self._predicted_allele_frequencies = (
            np.zeros(len(node_counts.node_counts)) + 1.0
        )  # default is 1.0, we will only change variant nodes

    def get_node_count(self, node):
        return self._node_counts.node_counts[node]

    def expected_count_following_node(self, node):
        if self._node_count_model is None:
            return 1

        count = (
            self._average_coverage
            * self._node_count_model.node_counts_following_node[node]
        )
        count += 0.5
        return count

    def expected_count_not_following_node(self, node):
        if self._node_count_model is None:
            return 0.01

        count = (
            self._average_coverage
            * self._node_count_model.node_counts_not_following_node[node]
        )
        count += 0.025
        return count

    def prob_observing_counts(self, ref_node, variant_node, genotype, type="binomial"):
        ref_count = self.get_node_count(ref_node)
        variant_count = self.get_node_count(variant_node)

        if isinstance(self._node_count_model, GenotypeNodeCountModel):
            if genotype == "1/1":
                counts = self._node_count_model.counts_homo_alt
            elif genotype == "0/0":
                counts = self._node_count_model.counts_homo_ref
            elif genotype == "0/1":
                counts = self._node_count_model.counts_hetero
            else:
                raise Exception("Unsupported genotype %s" % genotype)

            expected_count_alt = counts[variant_node]
            expected_count_ref = counts[ref_node]

            # Add dummy counts
            """
            if genotype == "1/1":
                expected_count_alt = max(2, expected_count_alt)
                expected_count_ref = max(0.01, expected_count_ref)
            elif genotype == "0/0":
                expected_count_ref = max(2, expected_count_ref)
                expected_count_alt = max(0.01, expected_count_alt)
            elif genotype == "0/1":
                expected_count_ref = max(1, expected_count_ref)
                expected_count_alt = max(1, expected_count_alt)
            """
            if genotype == "1/1":
                expected_count_alt += 0.1
                expected_count_ref += 0.01
            elif genotype == "0/0":
                expected_count_ref += 0.1
                expected_count_alt += 0.01
            elif genotype == "0/1":
                expected_count_ref += 0.056
                expected_count_alt += 0.056

            if genotype != "1/1":
                assert (
                    expected_count_ref > 0
                ), "Expected count ref is 0 for node %d, genotype %s" % (
                    ref_node,
                    genotype,
                )

            if genotype != "0/0":
                assert expected_count_alt > 0

        else:
            # Haplotype node count model
            if genotype == "1/1":
                expected_count_ref = (
                    self.expected_count_not_following_node(ref_node) * 2
                )
                expected_count_alt = (
                    self.expected_count_following_node(variant_node) * 2
                )
            elif genotype == "0/0":
                expected_count_ref = self.expected_count_following_node(ref_node) * 2
                expected_count_alt = (
                    self.expected_count_not_following_node(variant_node) * 2
                )
            elif genotype == "0/1":
                expected_count_ref = self.expected_count_following_node(
                    ref_node
                ) + self.expected_count_not_following_node(ref_node)
                expected_count_alt = self.expected_count_not_following_node(
                    variant_node
                ) + self.expected_count_following_node(variant_node)
            else:
                raise Exception("Unsupported genotype %s" % genotype)

        if type == "binomial":
            p = expected_count_ref / (expected_count_alt + expected_count_ref)
            k = int(ref_count)
            n = ref_count + variant_count
            assert (
                p > 0.0
            ), "Prob in binom is zero. Should not happen. k=%d, n=%d, p=%.10f" % (
                k,
                n,
                p,
            )
            return binom.pmf(k, n, p)

        return poisson.pmf(int(ref_count), expected_count_ref) * poisson.pmf(
            int(variant_count), expected_count_alt
        )

    def _set_predicted_allele_frequency(
        self,
        ref_node,
        var_node,
        prob_homo_ref,
        prob_homo_alt,
        prob_hetero,
        predicted_genotype,
    ):

        # if max(prob_homo_ref, prob_homo_alt, prob_hetero) < 0.97:
        #    ref_f

        if predicted_genotype == "0/0":
            ref_frequency = 1  # prob_homo_ref
        elif predicted_genotype == "1/1":
            ref_frequency = 0  # 1 - prob_homo_alt
        elif predicted_genotype == "0/1":
            ref_frequency = 0.5
        else:
            raise Exception("Invalid genotype")

        alt_frequency = 1 - ref_frequency

        if np.isnan(ref_frequency) or np.isnan(alt_frequency):
            if predicted_genotype == "0/0":
                ref_frequency = 1.0
                alt_frequency = 0.0
            elif predicted_genotype == "1/1":
                ref_frequency = 0.0
                alt_frequency = 1.0
            else:
                ref_frequency = 0.5
                alt_frequency = 0.5

        self._predicted_allele_frequencies[ref_node] = ref_frequency
        self._predicted_allele_frequencies[var_node] = alt_frequency

    def _genotype_biallelic_variant(
        self,
        reference_node,
        variant_node,
        a_priori_homozygous_ref,
        a_priori_homozygous_alt,
        a_priori_heterozygous,
        debug=False,
    ):

        p_counts_given_homozygous_ref = self.prob_observing_counts(
            reference_node, variant_node, "0/0"
        )
        p_counts_given_homozygous_alt = self.prob_observing_counts(
            reference_node, variant_node, "1/1"
        )
        p_counts_given_heterozygous = self.prob_observing_counts(
            reference_node, variant_node, "0/1"
        )

        prob_posteriori_heterozygous = (
            a_priori_heterozygous * p_counts_given_heterozygous
        )
        prob_posteriori_homozygous_alt = (
            a_priori_homozygous_alt * p_counts_given_homozygous_alt
        )
        prob_posteriori_homozygous_ref = (
            a_priori_homozygous_ref * p_counts_given_homozygous_ref
        )

        sum_of_posteriori = (
            prob_posteriori_homozygous_ref
            + prob_posteriori_heterozygous
            + prob_posteriori_homozygous_alt
        )

        prob_posteriori_heterozygous /= sum_of_posteriori
        prob_posteriori_homozygous_alt /= sum_of_posteriori
        prob_posteriori_homozygous_ref /= sum_of_posteriori

        sum_of_probs = (
            prob_posteriori_homozygous_ref
            + prob_posteriori_homozygous_alt
            + prob_posteriori_heterozygous
        )

        if abs(sum_of_probs - 1.0) > 0.01:
            logging.warning("Probs do not sum to 1.0: Sum is %.5f" % sum_of_probs)

        # Minimum count for genotyping
        if self.get_node_count(variant_node) <= 0:
            predicted_genotype = "0/0"

        if (
            prob_posteriori_homozygous_ref > prob_posteriori_homozygous_alt
            and prob_posteriori_homozygous_ref > prob_posteriori_heterozygous
        ):
            predicted_genotype = "0/0"
        elif (
            prob_posteriori_homozygous_alt > prob_posteriori_heterozygous
            and prob_posteriori_homozygous_alt > 0.0
        ):
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

        self._set_predicted_allele_frequency(
            reference_node,
            variant_node,
            prob_posteriori_homozygous_ref,
            prob_posteriori_homozygous_alt,
            prob_posteriori_heterozygous,
            predicted_genotype,
        )
        return predicted_genotype

    def get_allele_frequencies_from_most_similar_previous_variant(self, variant_id):

        (
            orig_prob_homo_ref,
            orig_prob_homo_alt,
            orig_prob_hetero,
        ) = self._genotype_frequencies.get_frequencies_for_variant(variant_id)
        orig_prob_homo_ref = max(0.001, orig_prob_homo_ref)
        orig_prob_hetero = max(0.001, orig_prob_hetero)
        orig_prob_homo_alt = max(0.001, orig_prob_homo_alt)

        if self._most_similar_variant_lookup is None:
            return orig_prob_homo_ref, orig_prob_homo_alt, orig_prob_hetero

        most_similar = self._most_similar_variant_lookup.get_most_similar_variant(
            variant_id
        )
        if most_similar in self._genotypes_called_at_variant:
            most_similar_genotype = self._genotypes_called_at_variant[most_similar]
            # prob_same_genotype = self._most_similar_variant_lookup.prob_of_having_the_same_genotype_as_most_similar(variant_id)

            prob_homo_ref_given_prev = (
                self._genotype_transition_probs.get_transition_probability(
                    variant_id, most_similar_genotype, 1
                )
            )
            prob_homo_alt_given_prev = (
                self._genotype_transition_probs.get_transition_probability(
                    variant_id, most_similar_genotype, 2
                )
            )
            prob_hetero_given_prev = (
                self._genotype_transition_probs.get_transition_probability(
                    variant_id, most_similar_genotype, 3
                )
            )

            if (
                prob_homo_alt_given_prev == 0
                and prob_homo_ref_given_prev == 0
                and prob_hetero_given_prev == 0
            ):
                # no data, probably no individuals having that genotype, fallback to population probs
                return orig_prob_homo_ref, orig_prob_homo_alt, orig_prob_hetero

            """
            prob_homo_ref_given_prev = 0
            prob_homo_alt_given_prev = 0
            prob_hetero_given_prev = 0

            if most_similar_genotype == 1:
                prob_homo_ref_given_prev = self._most_similar_variant_lookup.prob_of_having_the_same_genotype_as_most_similar(variant_id)
            elif most_similar_genotype == 2:
                prob_homo_alt_given_prev = self._most_similar_variant_lookup.prob_of_having_the_same_genotype_as_most_similar(variant_id)
            elif most_similar_genotype == 3:
                prob_hetero_ref_given_prev = self._most_similar_variant_lookup.prob_of_having_the_same_genotype_as_most_similar(variant_id)
            """

            sum_of_probs_given_prev = (
                prob_homo_alt_given_prev
                + prob_homo_ref_given_prev
                + prob_hetero_given_prev
            )
            if abs(sum_of_probs_given_prev - 1) < 0.001 and False:
                logging.warning(
                    "Probs given prev variant does not sum to 1: %.5f. Prev variant: %d, this variant: %d. Predicted: %d"
                    % (
                        sum_of_probs_given_prev,
                        most_similar,
                        variant_id,
                        most_similar_genotype,
                    )
                )

            prob_prev_genotype_is_correct = 0.97
            if most_similar_genotype != 1:
                prob_prev_genotype_is_correct = 0.97

            prob_homo_ref = (
                prob_homo_ref_given_prev * prob_prev_genotype_is_correct
                + (1 - prob_prev_genotype_is_correct) * orig_prob_homo_ref
            )
            prob_homo_alt = (
                prob_homo_alt_given_prev * prob_prev_genotype_is_correct
                + (1 - prob_prev_genotype_is_correct) * orig_prob_homo_alt
            )
            prob_hetero = (
                prob_hetero_given_prev * prob_prev_genotype_is_correct
                + (1 - prob_prev_genotype_is_correct) * orig_prob_hetero
            )

            """
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
            """

            return prob_homo_ref, prob_homo_alt, prob_hetero
        else:
            return orig_prob_homo_ref, orig_prob_homo_alt, orig_prob_hetero

    def genotype(self):

        variant_id = -1
        for i, variant in enumerate(self._variants):
            if i % 1000 == 0 and i > 0:
                logging.info("%d variants genotyped" % i)

            variant_id += 1
            variant_id = variant.vcf_line_number
            # self._genotypes_called_at_variant.append(0)
            assert (
                "," not in variant.variant_sequence
            ), "Only biallelic variants are allowed. Line is not bialleleic"

            debug = False
            try:
                reference_node = self._variant_to_nodes.ref_nodes[variant_id]
                variant_node = self._variant_to_nodes.var_nodes[variant_id]
            except VariantNotFoundException:
                continue

            # Compute from actual node counts instead (these are from traversing the graph)
            # prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_haplotype_counts(reference_node, variant_node)
            (
                prob_homo_ref,
                prob_homo_alt,
                prob_hetero,
            ) = self.get_allele_frequencies_from_most_similar_previous_variant(
                variant_id
            )

            predicted_genotype = self._genotype_biallelic_variant(
                reference_node,
                variant_node,
                prob_homo_ref,
                prob_homo_alt,
                prob_hetero,
                debug,
            )
            # self.add_individuals_with_genotype(predicted_genotype, reference_node, variant_node)

            # print("%s\t%s" % ("\t".join(l[0:9]), predicted_genotype))
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
