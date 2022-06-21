from obgraph.variant_to_nodes import VariantToNodes
from obgraph import GenotypeFrequencies, MostSimilarVariantLookup
from obgraph.genotype_matrix import GenotypeMatrix
import numpy as np
np.random.seed(1)
from kage import NodeCounts
from kage.node_count_model import NodeCountModel
from obgraph.variants import VcfVariants, VcfVariant
from kage.node_count_model import NodeCountModelAdvanced
from kage.combination_model_genotyper import CombinationModelGenotyper
from kage.helper_index import make_helper_model_from_genotype_matrix,make_helper_model_from_genotype_matrix_and_node_counts
from kage.sampling_combo_model import LimitedFrequencySamplingComboModel

class Helper:
    def __init__(self):

        self.n_nodes = 8
        self.max_node_id = self.n_nodes+1
        self.n_variants = 4
        self.n_individuals = 30
        self.genotype_matrix = None

    def prepare(self):
        assert self.genotype_matrix is not None, "Set genotype matrix before preparing"

        self.variant_to_nodes = VariantToNodes(np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]))
        self.most_simliar_variant_lookup = MostSimilarVariantLookup(np.arange(self.n_variants)-1, np.ones(self.n_variants))
        self.node_count_model = NodeCountModelAdvanced(np.zeros(self.max_node_id), np.zeros(self.max_node_id), np.zeros(self.max_node_id), np.zeros((self.max_node_id, 5)), np.zeros(self.max_node_id, dtype=bool))
        self.sampling_count_models = [LimitedFrequencySamplingComboModel.create_naive(4), LimitedFrequencySamplingComboModel.create_naive(4)]
       # self.helper_variants, self.genotype_combo_matrix = make_helper_model_from_genotype_matrix(self.genotype_matrix, self.most_simliar_variant_lookup)
        self.helper_variants, self.genotype_combo_matrix = \
            make_helper_model_from_genotype_matrix_and_node_counts(self.genotype_matrix, self.node_count_model, self.variant_to_nodes)
        print("Combo matrix")
        print(self.genotype_combo_matrix)


        self.input_variants = VcfVariants(
            [
                VcfVariant(1, 1, "A", "T", type="SNP"),
                VcfVariant(1, 2, "A", "T", type="SNP"),
                VcfVariant(1, 3, "A", "T", type="SNP"),
                VcfVariant(1, 4, "A", "T", type="SNP"),
            ]
        )

    def _make_random_genotype_matrix(self):
        matrix = np.random.randint(1, 4, (self.n_variants, self.n_individuals))
        self.genotype_matrix = GenotypeMatrix(matrix)

    def _make_genotype_matrix_biased_towards_genotype(self, genotype):
        assert genotype in [0, 1, 2]
        matrix = np.zeros((self.n_variants, self.n_individuals)) + genotype
        self.genotype_matrix = GenotypeMatrix(matrix)

    def run_test_with_node_counts(self, node_counts):
        # reset genotypes
        for variant in self.input_variants:
            variant.set_genotype(0, is_numeric=True)

        genotyper = CombinationModelGenotyper(self.sampling_count_models, 0, self.n_variants - 1, self.variant_to_nodes,
                                              node_counts,
                                              helper_model=self.helper_variants, helper_model_combo=self.genotype_combo_matrix
                                              )

        genotyper.genotype_and_modify_variants(self.input_variants)
        print("Predicted genotypes: %s" % genotyper._predicted_genotypes)
        print(genotyper.marginal_probs)
        print(self.input_variants)

    def testcase_all_homo_alt(self):
        # Case 1
        self._make_random_genotype_matrix()
        self.prepare()
        node_counts = NodeCounts(np.array([0.0] + [0.0] * 4 + [3.0] * 4))
        self.run_test_with_node_counts(node_counts)
        for variant in self.input_variants:
            assert variant.genotype == "1/1"

    def testcase_all_equally_likely_but_higher_priors_for_one_genotype(self):
        for genotype_numeric, genotype in [(0, "0/0"), (2, "1/1"), (1, "0/1")]:
            self._make_genotype_matrix_biased_towards_genotype(genotype_numeric)
            print("GENOTYPE MATRIX")
            print(self.genotype_matrix.matrix)
            self.prepare()

            node_counts = NodeCounts(np.zeros(self.max_node_id+1))
            self.run_test_with_node_counts(node_counts)
            for variant in self.input_variants:
                print(variant)
                assert variant.genotype == genotype


tester = Helper()

def test1():
    tester.testcase_all_homo_alt()

def test2():
    tester.testcase_all_equally_likely_but_higher_priors_for_one_genotype()

