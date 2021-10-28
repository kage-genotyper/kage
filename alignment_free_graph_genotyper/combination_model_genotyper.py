import logging
from .genotyper import Genotyper
from scipy.stats import binom
import numpy as np
from .negative_binomial_model_clean import CombinationModel, CombinationModelBothAlleles
from .node_count_model import GenotypeNodeCountModel, NodeCountModelAlleleFrequencies, NodeCountModelAdvanced
from .combination_model2 import ComboModel
from .new_helper_model import HelperModel


class CombinationModelGenotyper(Genotyper):
    def __init__(self, node_count_model, min_variant_id, max_variant_id, variant_to_nodes, node_counts,
                 genotype_frequencies=None, most_similar_variant_lookup=None, variant_window_size=500,
                 avg_coverage=15, genotype_transition_probs=None, tricky_variants=None, use_naive_priors=False,
                 helper_model=None, helper_model_combo=None):

        self._min_variant_id = min_variant_id
        self._max_variant_id = max_variant_id
        self._node_count_model = node_count_model
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
        self._genotypes_called_at_variant = {}  # index is variant id, genotype is 1,2,3 (homo ref, homo alt, hetero)
        self._predicted_allele_frequencies = np.zeros(
            len(node_counts.node_counts)) + 1.0  # default is 1.0, we will only change variant nodes
        self._predicted_genotypes = np.zeros(max_variant_id - min_variant_id + 1, dtype=np.uint8)
        self._prob_correct = np.zeros(max_variant_id - min_variant_id + 1, dtype=np.float)
        self._dummy_count_having_variant = 0.1 * avg_coverage / 15
        self._dummy_counts_not_having_variant = 0.1 * avg_coverage / 15
        self._genotype_transition_probs = genotype_transition_probs
        self._tricky_variants = tricky_variants
        self._use_naive_priors = use_naive_priors
        self._haplotype_coverage = avg_coverage / 2
        self._estimated_mapped_haplotype_coverage = self._haplotype_coverage * 0.75 * 0.85
        self.marginal_probs = None
        self._helper_model = helper_model
        self._helper_model_combo_matrix = helper_model_combo

    def compute_marginal_probabilities(self):
        # find expected count ref, alt for the three different genotypes
        ref_nodes = self._variant_to_nodes.ref_nodes[self._min_variant_id:self._max_variant_id + 1]
        alt_nodes = self._variant_to_nodes.var_nodes[self._min_variant_id:self._max_variant_id + 1]

        # Get observed counts
        observed_ref_nodes = self._node_counts.get_node_count_array()[ref_nodes]
        observed_alt_nodes = self._node_counts.get_node_count_array()[alt_nodes]

        # marginal probs go into a matrix, each row is one genotype, columns are variants
        marginal_probs = np.zeros((3, len(ref_nodes)))
        model = self._node_count_model


        logging.info("Using advanced node count model")
        # One model for ref nodes and one for alt nodes
        models = [None, None]
        for i, nodes in enumerate([ref_nodes, alt_nodes]):
            models[i] = ComboModel.from_counts(self._estimated_mapped_haplotype_coverage, model.frequencies[nodes],
                                                                  model.frequencies_squared[nodes],
                                                                  model.has_too_many[nodes],
                                                                  # np.ones_like(model.has_too_many[nodes]),
                                                                  model.certain[nodes],
                                                                  model.frequency_matrix[nodes])

        combination_model_both = CombinationModelBothAlleles(models[0], models[1])
        logging.info("Creating helper model")
        print(self._helper_model.shape)
        print(self._helper_model_combo_matrix.shape)
        helper_model = HelperModel(combination_model_both, self._helper_model, self._helper_model_combo_matrix)
        for i, genotype in enumerate([2, 0, 1]):
            logging.debug("Computing marginal probs for genotypes %s using combination model" % genotype)
            #probabilities = combination_model_both.pmf(observed_ref_nodes, observed_alt_nodes, genotype)
            probabilities = helper_model.logpmf(observed_ref_nodes, observed_alt_nodes, genotype)
            marginal_probs[i] = probabilities
            logging.info(np.where(np.isnan(marginal_probs[i])))

        # detect cases where all probs are zero and convert nans to zeros
        logging.info("Number of nan values in marginal probs: %d" % np.sum(np.isnan(marginal_probs)))
        marginal_probs = np.nan_to_num(marginal_probs)

        print(marginal_probs[0:300,:])

        zero_threshold = -1000000
        """
        all_zero = np.where((marginal_probs[0, :] < zero_threshold) & (marginal_probs[1, :] < zero_threshold) & (
                    marginal_probs[2, :] < zero_threshold))[0]
        marginal_probs[:, all_zero] = 1 / 3
        logging.info("%d out of %d variants have only zero probs" % (len(all_zero), len(ref_nodes)))
        """

        if self._tricky_variants is not None:
            tricky_variants = self._tricky_variants[self._min_variant_id:self._max_variant_id + 1]
            logging.info("There are %d tricky variants that will be given 1/3 as marginal probabilities" % np.sum(
                tricky_variants))
            marginal_probs[:, np.nonzero(tricky_variants)] = np.log(1 / 3)

        logging.info("Marginal probs computed")
        self.marginal_probs = marginal_probs

    def _genotype_biallelic_variant(self, variant_id, a_priori_homozygous_ref, a_priori_homozygous_alt,
                                    a_priori_heterozygous):

        p_counts_given_homozygous_ref = self.marginal_probs[0, variant_id - self._min_variant_id]
        p_counts_given_homozygous_alt = self.marginal_probs[1, variant_id - self._min_variant_id]
        p_counts_given_heterozygous = self.marginal_probs[2, variant_id - self._min_variant_id]

        if self._helper_model is not None:
            prob_posteriori_heterozygous = p_counts_given_heterozygous
            prob_posteriori_homozygous_alt = p_counts_given_homozygous_alt
            prob_posteriori_homozygous_ref = p_counts_given_homozygous_ref
        else:
            prob_posteriori_heterozygous = a_priori_heterozygous + p_counts_given_heterozygous
            prob_posteriori_homozygous_alt = a_priori_homozygous_alt + p_counts_given_homozygous_alt
            prob_posteriori_homozygous_ref = a_priori_homozygous_ref + p_counts_given_homozygous_ref

            sum_of_posteriori = prob_posteriori_homozygous_ref + prob_posteriori_heterozygous + prob_posteriori_homozygous_alt

            prob_posteriori_heterozygous -= sum_of_posteriori
            prob_posteriori_homozygous_alt -= sum_of_posteriori
            prob_posteriori_homozygous_ref -= sum_of_posteriori

        """
        sum_of_probs = prob_posteriori_homozygous_ref + prob_posteriori_homozygous_alt + prob_posteriori_heterozygous

        if abs(sum_of_probs - 1.0) > 0.01:
            logging.warning("Probs do not sum to 1.0: Sum is %.5f" % sum_of_probs)
        """

        if prob_posteriori_homozygous_ref > prob_posteriori_homozygous_alt and prob_posteriori_homozygous_ref > prob_posteriori_heterozygous:
            predicted_genotype = "0/0"
            prob_correct = prob_posteriori_homozygous_ref
        elif prob_posteriori_homozygous_alt > prob_posteriori_heterozygous:  # and prob_posteriori_homozygous_alt > 0.0:
            predicted_genotype = "1/1"
            prob_correct = prob_posteriori_homozygous_alt
        else:
            predicted_genotype = "0/1"
            prob_correct = prob_posteriori_heterozygous

        return predicted_genotype, prob_correct

    def genotype(self):
        # self._min_variant_id = self._variants[0].vcf_line_number
        # self._max_variant_id = self._variants[-1].vcf_line_number
        logging.debug("Min variant id is %d" % self._min_variant_id)
        logging.debug("Max variant id is %d" % self._max_variant_id)
        self.compute_marginal_probabilities()

        # for i, variant in enumerate(self._variants):
        for i, variant_id in enumerate(range(self._min_variant_id, self._max_variant_id+1)):
            if i % 500000 == 0 and i > 0:
                logging.info("%d variants genotyped" % i)

            # variant_id = variant.vcf_line_number
            # self._genotypes_called_at_variant.append(0)
            # assert "," not in variant.variant_sequence, "Only biallelic variants are allowed. Line is not bialleleic"

            if self._helper_model is not None:
                # using helper model, not getting priors from genotype frequencies
                predicted_genotype, prob_correct = self._genotype_biallelic_variant(variant_id,
                                                                                    None, None, None)

            else:
                if self._use_naive_priors:
                    prob_homo_ref = self._genotype_frequencies.homo_ref[variant_id]
                    prob_homo_alt = self._genotype_frequencies.homo_alt[variant_id]
                    prob_hetero = self._genotype_frequencies.hetero[variant_id]
                else:
                    prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_most_similar_previous_variant(
                        variant_id)

                predicted_genotype, prob_correct = self._genotype_biallelic_variant(variant_id,
                                                                                    np.log(prob_homo_ref),
                                                                                    np.log(prob_homo_alt),
                                                                                    np.log(prob_hetero))

            # self.add_individuals_with_genotype(predicted_genotype, reference_node, variant_node)

            # print("%s\t%s" % ("\t".join(l[0:9]), predicted_genotype))
            # variant.set_genotype(predicted_genotype)
            numeric_genotype = 0
            if predicted_genotype == "0/0":
                numeric_genotype = 1
            elif predicted_genotype == "1/1":
                numeric_genotype = 2
            elif predicted_genotype == "0/1":
                numeric_genotype = 3

            self._predicted_genotypes[i] = numeric_genotype
            self._prob_correct[i] = prob_correct

            if self._most_similar_variant_lookup is not None:
                self._genotypes_called_at_variant[variant_id] = numeric_genotype

        return self._predicted_genotypes, self._prob_correct

    def genotype_and_modify_variants(self, variants):
        self.genotype()
        for i, genotype in enumerate(self._predicted_genotypes):
            variants[i].set_genotype(genotype, is_numeric=True)

        return self._predicted_genotypes, self._prob_correct