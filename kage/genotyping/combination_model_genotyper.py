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
from kage.indexing.tricky_variants import TrickyVariants


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
        self._tricky_alleles = index.tricky_alleles if hasattr(index, "tricky_alleles") else None
        if self._tricky_variants is not None:
            logging.info("Will use tricky alleles")
            self.index.count_model[0].set_tricky_alleles(self._tricky_alleles[0].tricky_variants)
            self.index.count_model[1].set_tricky_alleles(self._tricky_alleles[1].tricky_variants)

        self._predicted_genotypes = np.zeros(max_variant_id - min_variant_id + 1, dtype=np.uint8)
        self._prob_correct = np.zeros(max_variant_id - min_variant_id + 1, dtype=float)
        self._haplotype_coverage = self.config.avg_coverage / 2
        #self._estimated_mapped_haplotype_coverage = self._haplotype_coverage * 0.75 * 0.85
        self._estimated_mapped_haplotype_coverage = self._haplotype_coverage * 0.85
        self.marginal_probs = None

    def predict(self):
        # find expected count ref, alt for the three different genotypes
        n_node_counts = len(self._node_counts.get_node_count_array())
        log_memory_usage_now("Genotyping, before getting ref/var nodes")
        ref_nodes = self.index.variant_to_nodes.ref_nodes[
            self._min_variant_id : self._max_variant_id + 1
        ]
        alt_nodes = self.index.variant_to_nodes.var_nodes[
            self._min_variant_id : self._max_variant_id + 1
        ]
        log_memory_usage_now("Genotyping, before getting observed counts")

        # Get observed counts
        observed_ref_nodes = self._node_counts.get_node_count_array(min_nodes=ref_nodes[-1])[ref_nodes]
        observed_alt_nodes = self._node_counts.get_node_count_array(min_nodes=alt_nodes[-1])[alt_nodes]

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


def downscale_coverage(config, node_counts, base):
    """
    Downscales node counts and coverage. Used to not overestimate prob of genotypes when coverage is high.
    """
    factor = config.avg_coverage / base
    logging.info("Before scale: %s" % node_counts.node_counts)
    node_counts.node_counts = np.round(node_counts.node_counts / factor)
    logging.info("After scale: %s" % node_counts.node_counts)
    config.avg_coverage = base


def add_svs_to_tricky_variants(index):
    variants = index.vcf_variants
    is_sv = (variants.ref_seq.shape[1] > 50) | (variants.alt_seq.shape[1] >= 50)
    logging.info(f"Will ignore reads for {np.sum(is_sv)} SVs")
    index.tricky_variants.add(TrickyVariants(is_sv))


def set_uniform_probs_for_svs(variants, probs):
    is_sv = (variants.ref_seq.shape[1] > 50) | (variants.alt_seq.shape[1] >= 50)
    logging.info(f"Will set uniform probs for {np.sum(is_sv)} SVs")
    probs[is_sv, :] = np.log(1 / 3)
    return probs
