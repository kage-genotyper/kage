import logging
import time
import numpy as np
from kage.genotyping.genotyper import Genotyper
from kage.models.models import HelperModel, ComboModelBothAlleles, NoHelperModel
from shared_memory_wrapper import (
    from_shared_memory,
    SingleSharedArray,
)
from shared_memory_wrapper.shared_memory import object_from_shared_memory
from kage.node_counts import NodeCounts
from ..configuration import GenotypingConfig

genotypes = ["0/0", "1/1", "0/1"]
numeric_genotypes = [1, 2, 3]
internal_genotypes = [0, 2, 1]
internal2numeric = dict(zip(internal_genotypes, numeric_genotypes))
numeric2internal = dict(zip(numeric_genotypes, internal_genotypes))


def translate_to_numeric(internal_genotypes, out=None):
    if out is None:
        out = np.empty_like(internal_genotypes)
    for k, v in internal2numeric.items():
        out[internal_genotypes == k] = v
    return out


class CombinationModelGenotyper(Genotyper):
    def __init__(
        self,
        count_models,
        min_variant_id,
        max_variant_id,
        variant_to_nodes,
        node_counts,
        genotype_frequencies=None,
        tricky_variants=None,
        helper_model=None,
        helper_model_combo=None,
        config=None
        #use_naive_priors=False,
        #n_threads=8,
        #avg_coverage=15,
        #ignore_helper_model=False,
        #ignore_helper_variants=False,
    ):

        self.config = config if config is not None else GenotypingConfig()  # GenotypingConfig(avg_coverage, use_naive_priors, n_threads, ignore_helper_model, ignore_helper_variants)


        self._min_variant_id = min_variant_id
        self._max_variant_id = max_variant_id
        self._count_models = count_models
        self._genotype_frequencies = genotype_frequencies
        self._variant_to_nodes = variant_to_nodes
        self._node_counts = node_counts
        self.expected_read_error_rate = 0.03
        #self._average_coverage = 7.0
        self._average_node_count_followed_node = 7.0
        self._individuals_with_genotypes = []
        self._variant_counter = 0
        self._genotypes_called_at_variant = (
            {}
        )  # index is variant id, genotype is 1,2,3 (homo ref, homo alt, hetero)
        self._predicted_allele_frequencies = (
            np.zeros(len(node_counts.node_counts)) + 1.0
        )  # default is 1.0, we will only change variant nodes
        self._predicted_genotypes = np.zeros(
            max_variant_id - min_variant_id + 1, dtype=np.uint8
        )
        self._prob_correct = np.zeros(max_variant_id - min_variant_id + 1, dtype=float)
        self._tricky_variants = tricky_variants
        #self._use_naive_priors = use_naive_priors
        self._haplotype_coverage = self.config.avg_coverage / 2
        self._estimated_mapped_haplotype_coverage = (
            self._haplotype_coverage * 0.75 * 0.85
        )
        self.marginal_probs = None
        self._helper_model = helper_model
        self._helper_model_combo_matrix = helper_model_combo
        #self._n_threads = n_threads
        #self._ignore_helper_model = ignore_helper_model
        #self._ignore_helper_variants = ignore_helper_variants

    @staticmethod
    def make_combomodel_from_shared_memory_data(data):
        logging.info("Starting creating combomodel in sepearte thread")
        t_start = time.perf_counter()
        (from_variant, to_variant), data = data
        (
            count_model_name,
            ref_nodes_name,
            alt_nodes_name,
            node_counts_name,
            self.config.avg_coverage,
        ) = data

        node_counts = from_shared_memory(NodeCounts, node_counts_name)
        ref_nodes = from_shared_memory(SingleSharedArray, ref_nodes_name).array
        alt_nodes = from_shared_memory(SingleSharedArray, alt_nodes_name).array
        count_models = object_from_shared_memory(count_model_name)

        observed_ref_nodes = node_counts.get_node_count_array()[ref_nodes]
        observed_alt_nodes = node_counts.get_node_count_array()[alt_nodes]

        t = time.perf_counter()
        models = [c.slice(from_variant, to_variant) for c in count_models]
        model_both_alleles = ComboModelBothAlleles(*models)
        logging.info(
            "Time spent on thread making models: %.3f" % (time.perf_counter() - t)
        )
        t = time.perf_counter()
        model_both_alleles.compute_logpmfs(
            observed_ref_nodes[from_variant:to_variant],
            observed_alt_nodes[from_variant:to_variant],
        )
        logging.info(
            "Time spent on thread computing logpmfs: %.3f" % (time.perf_counter() - t)
        )

        model_both_alleles.clear()  # remove data we don't need to make pickling faster
        logging.info(
            "Time spent making combomodel one thread: %.4f" % (time.perf_counter() - t)
        )
        return model_both_alleles

    def predict(self):
        # find expected count ref, alt for the three different genotypes
        ref_nodes = self._variant_to_nodes.ref_nodes[
            self._min_variant_id : self._max_variant_id + 1
        ]
        alt_nodes = self._variant_to_nodes.var_nodes[
            self._min_variant_id : self._max_variant_id + 1
        ]

        # Get observed counts
        observed_ref_nodes = self._node_counts.get_node_count_array()[ref_nodes]
        observed_alt_nodes = self._node_counts.get_node_count_array()[alt_nodes]

        # One model for ref nodes and one for alt nodes
        logging.info("Creating combomodels")

        combination_model_both = ComboModelBothAlleles(*self._count_models)

        if self.config.ignore_helper_model:
            logging.info("Ignoring helper model! Will not use helper variants to improve genotype accuracy")
            final_model = combination_model_both
        elif self.config.ignore_helper_variants:
            logging.info("Using NoHelperModel")
            final_model = NoHelperModel(combination_model_both,
                                        self._genotype_frequencies,
                                        self._tricky_variants,
                                        self._estimated_mapped_haplotype_coverage
                                        )
        else:
            final_model = HelperModel(
                combination_model_both,
                self._helper_model,
                self._helper_model_combo_matrix,
                self._tricky_variants,
                self._estimated_mapped_haplotype_coverage,
                ignore_helper_variants=self.config.ignore_helper_variants
            )

        genotypes, probabilities = final_model.predict(
            observed_ref_nodes, observed_alt_nodes, return_probs=True
        )
        logging.info("Translating genotypes to numeric")
        self._predicted_genotypes = translate_to_numeric(genotypes)
        self._count_probs = final_model.count_probs
        self._prob_correct = probabilities
        

    def genotype(self):
        self.predict()
        return self._predicted_genotypes, self._prob_correct, self._count_probs

    def genotype_and_modify_variants(self, variants):
        self.genotype()
        for i, genotype in enumerate(self._predicted_genotypes):
            variants[i].set_genotype(genotype, is_numeric=True)

        return self._predicted_genotypes, self._prob_correct
