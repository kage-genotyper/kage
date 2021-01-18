from .chain_mapper import ChainMapper
import numpy as np
from .letter_sequence_to_numeric import letter_sequence_to_numeric
from Bio.Seq import Seq

def get_power_array(k):
    return np.power(4, np.arange(0, k))


def read_kmers(read, power_array=None):
    numeric = letter_sequence_to_numeric(read)
    return np.convolve(numeric, power_array, mode='valid')  # % 452930477


class SimpleBestChainFinder:
    def __init__(self, graph_kmer_index, k=31):
        self._kmer_index = graph_kmer_index
        self._k = k
        self._power_array = np.power(4, np.arange(0, self._k))

    def _get_read_chains_only_one_direction(self, read):
        kmers = read_kmers(read, self._power_array)
        nodes, ref_offsets, read_offsets, frequencies = self._kmer_index.get_nodes_and_ref_offsets_from_multiple_kmers(
            kmers, max_hits=100)

        if len(nodes) == 0:
            return []
        chains = ChainMapper.find_chains(ref_offsets, read_offsets, nodes, frequencies, kmers=kmers)
        return chains


    def get_nodes_in_best_chain_for_read(self, read):
        chains = self._get_read_chains_only_one_direction(read)
        reverse_chains = self._get_read_chains_only_one_direction(str(Seq(read).reverse_complement()))
        chains.extend(reverse_chains)
        chains = sorted(chains, key=lambda c: c[2], reverse=True)
        if len(chains) == 0:
            return np.array([])

        best_chain = chains[0]
        return best_chain[1]
