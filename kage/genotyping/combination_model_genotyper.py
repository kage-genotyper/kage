import logging
import math
import time
import numpy as np
from kage.models.models import HelperModel, ComboModelBothAlleles, NoHelperModel
from shared_memory_wrapper import (
    from_shared_memory,
    SingleSharedArray,
)
from shared_memory_wrapper import object_from_shared_memory
from kage.node_counts import NodeCounts
from ..configuration import GenotypingConfig
from ..util import log_memory_usage_now

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


class CombinationModelGenotyper:
    def __init__( self, min_variant_id: int, max_variant_id: int, node_counts, index, config=None):

        self.config = config if config is not None else GenotypingConfig()
        self.index = index

        self._min_variant_id = min_variant_id
        self._max_variant_id = max_variant_id
        self._node_counts = node_counts

        self._tricky_variants = index.tricky_variants.tricky_variants if hasattr(index, "tricky_variants") else None
        self._predicted_genotypes = np.zeros(max_variant_id - min_variant_id + 1, dtype=np.uint8)
        self._prob_correct = np.zeros(max_variant_id - min_variant_id + 1, dtype=float)
        self._haplotype_coverage = self.config.avg_coverage / 2
        self._estimated_mapped_haplotype_coverage = self._haplotype_coverage * 0.75 * 0.85
        self.marginal_probs = None

    def predict(self):
        # find expected count ref, alt for the three different genotypes
        log_memory_usage_now("Genotyping, before getting ref/var nodes")
        ref_nodes = self.index.variant_to_nodes.ref_nodes[
            self._min_variant_id : self._max_variant_id + 1
        ]
        alt_nodes = self.index.variant_to_nodes.var_nodes[
            self._min_variant_id : self._max_variant_id + 1
        ]
        log_memory_usage_now("Genotyping, before getting observed counts")

        # Get observed counts
        observed_ref_nodes = self._node_counts.get_node_count_array()[ref_nodes]
        observed_alt_nodes = self._node_counts.get_node_count_array()[alt_nodes]

        # One model for ref nodes and one for alt nodes
        logging.info("Creating combomodels")

        log_memory_usage_now("Before model")
        combination_model_both = ComboModelBothAlleles(*self.index.count_model)
        log_memory_usage_now("After model")

        if self.config.ignore_helper_model:
            logging.info("Ignoring helper model! Will not use helper variants to improve genotype accuracy")
            final_model = combination_model_both
        elif self.config.ignore_helper_variants:
            assert False, "Not supported now"
            #logging.info("Using NoHelperModel")
            #final_model = NoHelperModel(combination_model_both,
            #                            self._genotype_frequencies,
            #                            self._tricky_variants,
            #                            self._estimated_mapped_haplotype_coverage
            #                            )
        else:
            final_model = HelperModel(
                combination_model_both,
                self.index.helper_variants.helper_variants,
                self.index.combination_matrix.matrix,
                self._tricky_variants,
                self._estimated_mapped_haplotype_coverage,
                ignore_helper_variants=self.config.ignore_helper_variants,
                gpu=self.config.use_gpu,
                n_threads=self.config.n_threads
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
        if self.config.min_genotype_quality > 0:
            set_to_homo_ref = math.e ** np.max(self._prob_correct, axis=1) < self.config.min_genotype_quality
            logging.warning("%d genotypes have lower prob than %.4f. Setting these to homo ref."
                            % (np.sum(set_to_homo_ref), self.config.min_genotype_quality))
            self._predicted_genotypes[set_to_homo_ref] = 0

        return self._predicted_genotypes, self._prob_correct, self._count_probs

    def genotype_and_modify_variants(self, variants):
        self.genotype()
        for i, genotype in enumerate(self._predicted_genotypes):
            variants[i].set_genotype(genotype, is_numeric=True)

        return self._predicted_genotypes, self._prob_correct
