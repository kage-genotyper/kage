from .node_counts import NodeCounts
from obgraph.variant_to_nodes import VariantToNodes
from obgraph import GenotypeFrequencies, MostSimilarVariantLookup
from .node_count_model import NodeCountModel, GenotypeNodeCountModel
import numpy as np
from .variants import VcfVariant, VcfVariants
from collections import defaultdict
from .helper_index import make_helper_model_from_genotype_matrix
from obgraph.genotype_matrix import GenotypeMatrix


def run_genotyper_on_simualated_data(genotyper, n_variants, average_coverage, coverage_std, duplication_rate):
    simulator = GenotypingDataSimulator(n_variants, average_coverage, coverage_std, duplication_rate)
    variants, node_counts, model, most_similar_variant_lookup, variant_to_nodes, helper_model, helper_model_combo_matrix = simulator.run()
    truth_variants = variants.copy()

    print(node_counts)
    #g = genotyper(model, variants, variant_to_nodes, node_counts, genotype_frequencies, most_similar_variant_lookup)
    g = genotyper(model, 0, len(variants)-1, variant_to_nodes, node_counts, None,
                    most_similar_variant_lookup, avg_coverage=1,
                    helper_model=helper_model, helper_model_combo=helper_model_combo_matrix
                    )
    g.genotype_and_modify_variants(variants)

    truth_variants.compute_similarity_to_other_variants(variants)

    print("")
    print("Genotyping accuracy: %.4f " % truth_variants.compute_similarity_to_other_variants(variants))


class GenotypingDataSimulator:
    def __init__(self, n_variants, average_coverage=7, coverage_std=3, duplication_rate=0.1):
        self._n_variants = n_variants
        self._n_individuals = 100
        self._average_coverage = average_coverage
        self._coverage_std = coverage_std
        self._duplication_rate = duplication_rate

        n_nodes = self._n_variants*2+1
        self._n_nodes = n_nodes
        self._expected_counts_following_node = np.zeros(n_nodes)
        self._expected_counts_not_following_node = np.zeros(n_nodes)
        self._node_counts = np.zeros(n_nodes)
        self._variants = None

    def _make_random_genotype_matrix(self):
        matrix = np.random.randint(1, 4, (self._n_individuals, self._n_variants))
        self.genotype_matrix = GenotypeMatrix(matrix)

    def run(self):
        self._make_random_genotype_matrix()
        self._reference_nodes = np.arange(1, self._n_variants+1)
        self._variant_nodes = np.arange(self._n_variants+1, self._n_variants*2+1)
        variant_to_nodes = VariantToNodes(self._reference_nodes, self._variant_nodes)
        most_simliar_variant_lookup = MostSimilarVariantLookup(np.arange(1, self._n_variants+1)-1,
                                                               np.array([1.0] * self._n_variants))
        self.most_simliar_variant_lookup = MostSimilarVariantLookup(np.arange(self._n_variants) - 1,
                                                                    np.ones(self._n_variants))
        self.helper_variants, self.genotype_combo_matrix = make_helper_model_from_genotype_matrix(self.genotype_matrix,
                                                                                                  self.most_simliar_variant_lookup)

        self._simulate_variants()

        return self._variants, NodeCounts(self._node_counts), \
               self._model, most_simliar_variant_lookup, variant_to_nodes, \
               self.helper_variants, self.genotype_combo_matrix


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
                else:
                    ref_count += real_reads_on_haplotype

            self._node_counts[self._variant_nodes[i]] += var_count
            self._node_counts[self._reference_nodes[i]] += ref_count

            variants.append(variant)


        # Duplications
        n_duplications_per_variant_ref = defaultdict(int)
        n_duplications_per_variant_variant = defaultdict(int)

        duplicate_frequencies = defaultdict(list)  # node id to allele frequency at duplicate


        for i in range(self._n_variants):
            if np.random.random() < self._duplication_rate:
                n_duplicate_areas = np.random.randint(0, 3)
                for duplicate_area in range(n_duplicate_areas):
                    reads_from_duplicate = int(np.random.normal(self._average_coverage, self._coverage_std))
                    # 50/50 chance that this duplicate is on variant or ref node
                    if np.random.random() < 0.5:
                        self._expected_counts_following_node[self._variant_nodes[i]] += self._average_coverage
                        self._expected_counts_not_following_node[self._variant_nodes[i]] += self._average_coverage
                        n_duplications_per_variant_variant[i] += 1
                        duplicate_frequencies[self._variant_nodes[i]].append(0.5)
                    else:
                        self._expected_counts_following_node[self._reference_nodes[i]] += self._average_coverage
                        self._expected_counts_not_following_node[self._reference_nodes[i]] += self._average_coverage
                        n_duplications_per_variant_ref[i] += 1
                        duplicate_frequencies[self._reference_nodes[i]].append(0.5)


        # Add observed counts from duplicates
        for variant_id, n_duplications in n_duplications_per_variant_ref.items():
            for i in range(n_duplications):
                self._node_counts[self._reference_nodes[variant_id]] += int(np.random.normal(self._average_coverage, self._coverage_std))

        for variant_id, n_duplications in n_duplications_per_variant_variant.items():
            for i in range(n_duplications):
                self._node_counts[self._variant_nodes[variant_id]] += int(np.random.normal(self._average_coverage,
                                                                                         self._coverage_std))

        # non-duplicate counts, from actual variant
        for i in range(self._n_variants):
            self._expected_counts_following_node[self._reference_nodes[i]] += self._average_coverage
            self._expected_counts_following_node[self._variant_nodes[i]] += self._average_coverage

        # make model
        print(duplicate_frequencies)
        from .node_count_model import NodeCountModelAdvanced
        self._model = NodeCountModelAdvanced.from_dict_of_frequencies(duplicate_frequencies, self._n_nodes)

        # make helper model



        self._variants = VcfVariants(variants)
