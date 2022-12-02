import logging
import numpy as np


class VariantKmerAnalyser:
    def __init__(
        self,
        reverse_kmer_index,
        kmer_index,
        variant_to_nodes,
        write_good_variants_to_file,
    ):
        self.reverse_kmer_index = reverse_kmer_index
        self.kmer_index = kmer_index
        self.variant_to_nodes = variant_to_nodes
        self._n_good_variants = 0
        self._write_good_variants_to_file = write_good_variants_to_file
        self._good_variant_ids = np.zeros(
            len(self.variant_to_nodes.var_nodes + 1), dtype=np.uint8
        )
        self._n_supergood_variants = 0

    def _analyse_variant(self, variant_id):
        ref_node = self.variant_to_nodes.ref_nodes[variant_id]
        var_node = self.variant_to_nodes.var_nodes[variant_id]

        kmers = list(self.reverse_kmer_index.get_node_kmers(ref_node)) + list(
            self.reverse_kmer_index.get_node_kmers(var_node)
        )
        kmers = [int(kmer) for kmer in kmers]

        frequencies = [self.kmer_index.get_frequency(kmer) for kmer in kmers] + [0]

        if len(frequencies) <= 1:
            logging.warning("Few kmers for variant %d" % variant_id)

        if max(frequencies) <= 1:
            self._n_good_variants += 1

            if len(kmers) == 2:
                self._good_variant_ids[variant_id] = 1
                self._n_supergood_variants += 1

    def analyse(self):
        n_variants = len(self.variant_to_nodes.var_nodes)
        for variant_id in range(0, n_variants):
            if variant_id % 1000 == 0:
                logging.info("%d/%d processed" % (variant_id, n_variants))
            self._analyse_variant(variant_id)

        logging.info("N good variants: %d" % self._n_good_variants)
        logging.info("N supergood variants: %d" % self._n_supergood_variants)
        logging.info("Ratio of good: %.4f" % (self._n_good_variants / n_variants))

        if self._write_good_variants_to_file is not None:
            np.save(self._write_good_variants_to_file, self._good_variant_ids)
            logging.info(
                "Wrote good variants to file %s" % self._write_good_variants_to_file
            )
