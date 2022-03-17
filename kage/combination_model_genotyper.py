import logging
import time
from itertools import repeat
import numpy as np
from scipy.stats import binom
from .util import log_memory_usage_now
from .genotyper import Genotyper
from .node_count_model import GenotypeNodeCountModel, NodeCountModelAlleleFrequencies, NodeCountModelAdvanced
from .combomodel import ComboModel, ComboModelWithIncreasedZeroProb
from .models import HelperModel, ComboModelBothAlleles, ChunkedComboModelBothAlleles
from shared_memory_wrapper import to_shared_memory, from_shared_memory, SingleSharedArray
from shared_memory_wrapper.shared_memory import get_shared_pool
from pathos.multiprocessing import Pool
from .node_counts import NodeCounts


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
                 helper_model=None, helper_model_combo=None, n_threads=8):

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
        self._prob_correct = np.zeros(max_variant_id - min_variant_id + 1, dtype=float)
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
        self._n_threads = n_threads

    @staticmethod
    def make_combomodel_from_shared_memory_data(data):
        logging.info("Starting creating combomodel in sepearte thread")
        t_start = time.perf_counter()
        (from_variant, to_variant), data = data
        node_count_model_name, ref_nodes_name, alt_nodes_name, node_counts_name, avg_coverage = data

        node_counts = from_shared_memory(NodeCounts, node_counts_name)
        ref_nodes = from_shared_memory(SingleSharedArray, ref_nodes_name).array
        alt_nodes = from_shared_memory(SingleSharedArray, alt_nodes_name).array
        node_count_model = from_shared_memory(NodeCountModelAdvanced, node_count_model_name)

        observed_ref_nodes = node_counts.get_node_count_array()[ref_nodes]
        observed_alt_nodes = node_counts.get_node_count_array()[alt_nodes]

        t = time.perf_counter()
        models = [ComboModel.from_counts(avg_coverage,
                                         node_count_model.frequencies[nodes],
                                         node_count_model.frequencies_squared[nodes],
                                         node_count_model.has_too_many[nodes],
                                         node_count_model.certain[nodes],
                                         node_count_model.frequency_matrix[nodes])
                  for nodes in (ref_nodes[from_variant:to_variant], alt_nodes[from_variant:to_variant])]
        model_both_alleles = ComboModelBothAlleles(*models)
        logging.info("Time spent on thread making models: %.3f" % (time.perf_counter()-t))
        t = time.perf_counter()
        model_both_alleles.compute_logpmfs(observed_ref_nodes[from_variant:to_variant],
                                           observed_alt_nodes[from_variant:to_variant])
        logging.info("Time spent on thread computing logpmfs: %.3f" % (time.perf_counter()-t))

        model_both_alleles.clear()  # remove data we don't need to make pickling faster
        logging.info("Time spent making combomodel one thread: %.4f" % (time.perf_counter()-t))
        return model_both_alleles

    def predict(self):
        # find expected count ref, alt for the three different genotypes
        ref_nodes = self._variant_to_nodes.ref_nodes[self._min_variant_id:self._max_variant_id + 1]
        alt_nodes = self._variant_to_nodes.var_nodes[self._min_variant_id:self._max_variant_id + 1]

        # Get observed counts
        observed_ref_nodes = self._node_counts.get_node_count_array()[ref_nodes]
        observed_alt_nodes = self._node_counts.get_node_count_array()[alt_nodes]

        node_count_model = self._node_count_model
        # One model for ref nodes and one for alt nodes
        start_time = time.perf_counter()
        logging.info("Creating combomodels")

        n_threads = 4
        max_variant_id = len(ref_nodes)
        variant_chunks = list([int(i) for i in np.linspace(0, max_variant_id, self._n_threads+1)])
        variant_chunks = [(from_pos, to_pos) for from_pos, to_pos in zip(variant_chunks[0:-1], variant_chunks[1:])]
        logging.info("Variant chunks: %s" % variant_chunks)

        node_count_model_name = to_shared_memory(node_count_model)
        ref_nodes_name = to_shared_memory(SingleSharedArray(ref_nodes))
        alt_nodes_name = to_shared_memory(SingleSharedArray(alt_nodes))
        node_counts_name = to_shared_memory(self._node_counts)
        logging.info("Done writing to shared memory")

        models_both_alleles = []
        logging.info("Making pool")
        pool = get_shared_pool(n_threads)
        logging.info("Done making pool")
        data = (node_count_model_name, ref_nodes_name, alt_nodes_name,  node_counts_name, self._estimated_mapped_haplotype_coverage)
        for model_both_alleles in pool.imap(CombinationModelGenotyper.make_combomodel_from_shared_memory_data,
                                             zip(variant_chunks, repeat(data))):
            models_both_alleles.append(model_both_alleles)

        logging.info("Done creating combomodels")

        t = time.perf_counter()
        combination_model_both = ChunkedComboModelBothAlleles(models_both_alleles)
        logging.info("Time concatenating into chunked combo model: %.4f" % (time.perf_counter()-t))


        helper_model = HelperModel(combination_model_both, self._helper_model, self._helper_model_combo_matrix, self._tricky_variants)
        genotypes, probabilities = helper_model.predict(observed_ref_nodes, observed_alt_nodes, return_probs=True)
        logging.info("Translating genotypes to numeric")
        self._predicted_genotypes = translate_to_numeric(genotypes)
        self._count_probs = helper_model.count_probs

        #min_alt_node_counts = 0
        #too_low_alt_counts = np.where(observed_alt_nodes < min_alt_node_counts)[0]
        #logging.info("There are %d variants with alt node counts less than minimum required (%d). Genotyping these as homo ref." % (len(too_low_alt_counts), min_alt_node_counts))
        #self._predicted_genotypes[too_low_alt_counts] = 0

        self._prob_correct = probabilities

    def genotype(self):
        self.predict()
        return self._predicted_genotypes, self._prob_correct, self._count_probs

    def genotype_and_modify_variants(self, variants):
        self.genotype()
        for i, genotype in enumerate(self._predicted_genotypes):
            variants[i].set_genotype(genotype, is_numeric=True)

        return self._predicted_genotypes, self._prob_correct
