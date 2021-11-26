import logging
import time

import numpy as np
from scipy.stats import binom
from .util import log_memory_usage_now
from .genotyper import Genotyper
from .node_count_model import GenotypeNodeCountModel, NodeCountModelAlleleFrequencies, NodeCountModelAdvanced
from .combomodel import ComboModel, ComboModelWithIncreasedZeroProb
from .models import HelperModel, ComboModelBothAlleles

genotypes = ["0/0", "1/1", "0/1"]
numeric_genotypes = [1, 2, 3]
internal_genotypes = [0, 2, 1]
internal2numeric = dict(zip(internal_genotypes, numeric_genotypes))
numeric2internal = dict(zip(numeric_genotypes, internal_genotypes))

def translate_to_numeric(internal_genotypes, out=None):
    if out is None:
        out = np.empty_like(internal_genotypes)
    for k, v in internal2numeric.items():
        out[internal_genotypes==k] = v
    return out

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

    def predict(self):
        # find expected count ref, alt for the three different genotypes
        ref_nodes = self._variant_to_nodes.ref_nodes[self._min_variant_id:self._max_variant_id + 1]
        alt_nodes = self._variant_to_nodes.var_nodes[self._min_variant_id:self._max_variant_id + 1]

        # Get observed counts
        observed_ref_nodes = self._node_counts.get_node_count_array()[ref_nodes]
        observed_alt_nodes = self._node_counts.get_node_count_array()[alt_nodes]

        model = self._node_count_model
        # One model for ref nodes and one for alt nodes
        start_time = time.perf_counter()
        models = [ComboModel.from_counts(self._estimated_mapped_haplotype_coverage, model.frequencies[nodes],
                                         model.frequencies_squared[nodes],
                                         model.has_too_many[nodes],
                                         model.certain[nodes],
                                         model.frequency_matrix[nodes])
                  for nodes in (ref_nodes, alt_nodes)]

        combination_model_both = ComboModelBothAlleles(*models)
        helper_model = HelperModel(combination_model_both, self._helper_model, self._helper_model_combo_matrix, self._tricky_variants)
        genotypes, probabilities = helper_model.predict(observed_ref_nodes, observed_alt_nodes, return_probs=True)
        self._predicted_genotypes = translate_to_numeric(genotypes)
        self._count_probs = helper_model.count_probs

        min_alt_node_counts = 0
        too_low_alt_counts = np.where(observed_alt_nodes < min_alt_node_counts)[0]
        logging.info("There are %d variants with alt node counts less than minimum required (%d). Genotyping these as homo ref." % (len(too_low_alt_counts), min_alt_node_counts))
        self._predicted_genotypes[too_low_alt_counts] = 0

        self._prob_correct = probabilities

    def genotype(self):
        self.predict()
        return self._predicted_genotypes, self._prob_correct, self._count_probs

    def genotype_and_modify_variants(self, variants):
        self.genotype()
        for i, genotype in enumerate(self._predicted_genotypes):
            variants[i].set_genotype(genotype, is_numeric=True)

        return self._predicted_genotypes, self._prob_correct
