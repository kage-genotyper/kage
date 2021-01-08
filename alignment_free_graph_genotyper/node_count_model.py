import numpy as np
from .simple_best_chain_finder import SimpleBestChainFinder
from Bio.Seq import Seq

class NodeCountModel:
    def __init__(self, node_counts_following_node, node_counts_not_following_node, average_coverage=1):
        self.node_counts_following_node = node_counts_following_node
        self.node_counts_not_following_node = node_counts_not_following_node

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(data["node_counts_following_node"], data["node_counts_not_following_node"])

    def to_file(self, file_name):
        np.savez(file_name, node_counts_following_node=self.node_counts_following_node,
            node_counts_not_following_node=self.node_counts_not_following_node)


class NodeCountModelCreatorFromSimpleChaining:
    def __init__(self, nodes_followed_by_individual, individual_genome_sequence, kmer_index, n_nodes, n_reads_to_simulate=1000, read_length=150):
        self.kmer_index = kmer_index
        self.nodes_followed_by_individual = nodes_followed_by_individual
        self.genome_sequence = individual_genome_sequence
        self.reverse_genome_sequence = str(Seq(self.genome_sequence).reverse_complement())
        self._node_counts_following_node = np.zeros(n_nodes, dtype=np.uint32)
        self._node_counts_not_following_node = np.zeros(n_nodes, dtype=np.uint32)
        self.n_reads_to_simulate = n_reads_to_simulate
        self.read_length = read_length
        self.genome_size = len(self.genome_sequence)

    def get_node_counts(self):
        # Simulate reads from the individual
        # for each read, find nodes in best chain
        # increase those node counts
        chain_finder = SimpleBestChainFinder(self.kmer_index)

        for i in range(0, self.n_reads_to_simulate):
            pos_start = np.random.randint(0, self.genome_size - self.read_length)
            pos_end = pos_start + self.genome_size

            for read in (self.genome_sequence[pos_start:pos_end], self.reverse_genome_sequence[pos_start:pos_end]):
                nodes = chain_finder.get_nodes_in_best_chain_for_read(read)
                for node in nodes:
                    if node in self.nodes_followed_by_individual:
                        self._node_counts_following_node[node] += 1
                    else:
                        self._node_counts_not_following_node[node] += 1

        return self._node_counts_following_node, self._node_counts_not_following_node