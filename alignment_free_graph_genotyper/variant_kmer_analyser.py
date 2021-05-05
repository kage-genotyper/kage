import logging
import numpy as np

class VariantKmerAnalyser:
    def __init__(self, reverse_kmer_index, kmer_index, variants, variant_to_nodes):
        self.reverse_kmer_index = reverse_kmer_index
        self.kmer_index = kmer_index
        self.variants = variants
        self.variant_to_nodes = variant_to_nodes
        self._n_good_variants = 0

    def _analyse_variant(self, variant):
        ref_node = self.variant_to_nodes.ref_nodes[variant.vcf_line_number]
        var_node = self.variant_to_nodes.var_nodes[variant.vcf_line_number]

        kmers = list(self.reverse_kmer_index.get_node_kmers(ref_node)) + list(self.reverse_kmer_index.get_node_kmers(var_node))
        kmers = [int(kmer) for kmer in kmers]

        frequencies = [self.kmer_index.get_frequency(kmer) for kmer in kmers] + [0]

        if len(frequencies) <= 2:
            logging.warning("Few kmers for variant %s" % variant)

        if max(frequencies) <= 3:
            self._n_good_variants += 1

    def analyse(self):
        for variant in self.variants:
            self._analyse_variant(variant)

        logging.info("N good variants: %d" % self._n_good_variants)
        logging.info("Ratio of good: %.4f" % (self._n_good_variants / len(self.variants)))
