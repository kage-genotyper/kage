from .node_counts import NodeCounts
from obgraph.variant_to_nodes import VariantToNodes
from obgraph import GenotypeFrequencies, MostSimilarVariantLookup
from .node_count_model import NodeCountModel
import numpy as np
from .variants import VcfVariant, VcfVariants


def run_genotyper_on_simualated_data(genotyper, n_variants, average_coverage, coverage_std):
    simulator = GenotypingDataSimulator(n_variants, average_coverage, coverage_std)
    variants, node_counts, model, genotype_frequencies, most_similar_variant_lookup, variant_to_nodes = simulator.run()

    truth_variants = variants.copy()

    g = genotyper(model, variants, variant_to_nodes, node_counts, genotype_frequencies, most_similar_variant_lookup)
    g.genotype()

    truth_variants.compute_similarity_to_other_variants(variants)

    print("")
    print("Gentyping accuracy: %.4f " % truth_variants.compute_similarity_to_other_variants(variants))


class GenotypingDataSimulator:
    def __init__(self, n_variants, average_coverage=7, coverage_std=3):
        self._n_variants = n_variants
        self._average_coverage = average_coverage
        self._coverage_std = coverage_std

        n_nodes = self._n_variants*2+1
        self._expected_counts_following_node = np.zeros(n_nodes)
        self._expected_counts_not_following_node = np.zeros(n_nodes)
        self._node_counts = np.zeros(n_nodes)
        self._variants = None

    def run(self):
        self._reference_nodes = np.arange(1, self._n_variants+1)
        self._variant_nodes = np.arange(self._n_variants+1, self._n_variants*2+1)
        variant_to_nodes = VariantToNodes(self._reference_nodes, self._variant_nodes)
        frequencies = np.array([1 / 3] * self._n_variants)
        genotype_frequencies = GenotypeFrequencies(frequencies, frequencies, frequencies)

        most_simliar_variant_lookup = MostSimilarVariantLookup(np.arange(1, self._n_variants+1)-1,
                                                               np.array([1.0] * self._n_variants))

        self._simulate_variants()

        return self._variants, NodeCounts(self._node_counts), NodeCountModel(self._expected_counts_following_node, self._expected_counts_not_following_node), \
                genotype_frequencies, most_simliar_variant_lookup, variant_to_nodes


    def _simulate_variants(self):
        variants = []
        possible_genotypes = ["0/0", "1/1", "0/1"]
        for i in range(self._n_variants):
            variant = VcfVariant(1, i, "A", "T", type="SNP")
            genotype = np.random.choice(possible_genotypes)
            variant.set_genotype(genotype)

            ref_count = 0
            var_count = 0

            for haplotype in genotype.split("/"):
                real_reads_on_haplotype = int(np.random.normal(self._average_coverage, self._coverage_std))

                if haplotype == "1":
                    var_count += real_reads_on_haplotype
                    self._expected_counts_following_node[self._variant_nodes[i]] += self._average_coverage
                else:
                    ref_count += real_reads_on_haplotype
                    self._expected_counts_following_node[self._reference_nodes[i]] += self._average_coverage

                if np.random.random() < 0.1:
                    n_duplicate_areas = np.random.randint(0, 3)
                    for duplicate_area in range(n_duplicate_areas):
                        reads_from_duplicate = int(np.random.normal(self._average_coverage, self._coverage_std))
                        # 50/50 chance that this duplicate follows the node or not
                        follows_node = False
                        if np.random.randint(0, 2) == 1:
                            follows_node = True

                        if haplotype == "1":
                            var_count += reads_from_duplicate
                            if follows_node:
                                self._expected_counts_following_node[self._variant_nodes[i]] += self._average_coverage
                            else:
                                self._expected_counts_not_following_node[self._variant_nodes[i]] += self._average_coverage
                        else:
                            ref_count += reads_from_duplicate
                            if follows_node:
                                self._expected_counts_following_node[self._reference_nodes[i]] += self._average_coverage
                            else:
                                self._expected_counts_not_following_node[self._reference_nodes[i]] += self._average_coverage


            variants.append(variant)
            self._node_counts[self._variant_nodes[i]] += var_count
            self._node_counts[self._reference_nodes[i]] += ref_count

        self._variants = VcfVariants(variants)
