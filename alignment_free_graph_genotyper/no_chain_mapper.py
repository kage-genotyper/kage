import numpy as np
from .chain_mapper import read_kmers, get_power_array
import logging
from Bio.Seq import Seq


class NoChainMapper:
    def __init__(self, reads, kmer_index, k=31, max_node_id=3000000):
        self._k = k
        self._reads = reads
        self._kmer_index = kmer_index
        self._node_counts = np.zeros(max_node_id+1, dtype=np.uint16)

    def get_counts(self):
        power_array = get_power_array(self._k)
        for i, read in enumerate(self._reads):
            if i % 20000 == 0:
                logging.info("%d reads processed." % i)

            for sequence in [read, str(Seq(read).reverse_complement())]:
                kmers = read_kmers(sequence, power_array)
                node_hits = self._kmer_index.get_nodes_from_multiple_kmers(kmers)
                if len(node_hits) > 0:
                    self._node_counts[node_hits] += 1

        return self._node_counts
