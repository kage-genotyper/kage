from obgraph.variant_to_nodes import VariantToNodes
from obgraph import GenotypeFrequencies, MostSimilarVariantLookup
import numpy as np
from alignment_free_graph_genotyper import NodeCounts
from alignment_free_graph_genotyper.node_count_model import NodeCountModel
from alignment_free_graph_genotyper.variants import GenotypeCalls, VariantGenotype
from alignment_free_graph_genotyper.statistical_node_count_genotyper import StatisticalNodeCountGenotyper

def test_simple():
    variant_to_nodes = VariantToNodes(np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]))
    frequencies = np.array([1/3] * 4)
    genotype_frequencies = GenotypeFrequencies(frequencies, frequencies, frequencies)
    most_simliar_variant_lookup = MostSimilarVariantLookup(np.array([0, 0, 1, 2, 3]), np.array([1.0, 1.0, 1.0, 1.0]))

    node_count_model = NodeCountModel(np.array([0.0] + [3.0] * 8), np.array([0.0] * 9))

    input_variants = GenotypeCalls(
        [
            VariantGenotype(1, 1, "A", "T", type="SNP"),
            VariantGenotype(1, 2, "A", "T", type="SNP"),
            VariantGenotype(1, 3, "A", "T", type="SNP"),
            VariantGenotype(1, 4, "A", "T", type="SNP"),
        ]
    )

    # Case 1
    node_counts = NodeCounts(np.array([0.0] + [0.0] * 4 + [3.0] * 4))
    genotyper = StatisticalNodeCountGenotyper(node_count_model, input_variants, variant_to_nodes, node_counts,
                                              genotype_frequencies, most_simliar_variant_lookup)

    genotyper.genotype()
    for variant in input_variants:
        assert variant.genotype == "1/1"

    # Case 2
    node_counts = NodeCounts(np.array([0.0] + [3.0] * 4 + [3.0] * 4))
    genotyper = StatisticalNodeCountGenotyper(node_count_model, input_variants, variant_to_nodes, node_counts,
                                              genotype_frequencies, most_simliar_variant_lookup)

    genotyper.genotype()
    for variant in input_variants:
        assert variant.genotype == "0/1"

    # Case 3
    node_counts = NodeCounts(np.array([0.0] + [3.0] * 4 + [0.0] * 4))
    genotyper = StatisticalNodeCountGenotyper(node_count_model, input_variants, variant_to_nodes, node_counts,
                                              genotype_frequencies, most_simliar_variant_lookup)

    genotyper.genotype()
    for variant in input_variants:
        assert variant.genotype == "0/0"

test_simple()