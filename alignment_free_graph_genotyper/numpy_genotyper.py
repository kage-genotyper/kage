from .genotyper import Genotyper

class NumpyGenotyper(Genotyper):
    def genotype(self):
        reference_nodes = self._variant_to_nodes.ref_nodes
        variant_nodes = self._variant_to_nodes.var_nodes
        expected_count_following_node = self._node_count_model.node_counts_following_node
        expected_count_not_following_node = self._node_count_model.node_counts_not_following_node
        node_counts = self._node_counts.get_node_count_array()

        genotype_frequencies = self._genotype_frequencies
        frequency_homo_ref = genotype_frequencies.homo_ref
        frequency_homo_alt = genotype_frequencies.homo_alt
        frequency_hetero = genotype_frequencies.hetero

        most_similar_variant_lookup = self._most_similar_variant_lookup

        for i, variant in enumerate(self._variants):

            if node_counts[variant_nodes[i]] > 10:
                variant.set_genotype("1/1")
            else:
                variant.set_genotype("0/0")

