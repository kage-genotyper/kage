import numpy as np
from Bio.Seq import Seq
import logging
from obgraph import Graph, VariantNotFoundException
from graph_kmer_index import ReverseKmerIndex, KmerIndex

from alignment_free_graph_genotyper import cython_chain_genotyper

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


class NodeCountModelCreatorFromNoChaining:
    def __init__(self, kmer_index: KmerIndex, reverse_index: ReverseKmerIndex, graph: Graph, variants, max_node_id):
        self.kmer_index = kmer_index
        self.graph = graph
        self.reverse_index = reverse_index
        self.variants = variants

        self.node_counts_following_node = np.zeros(max_node_id+1, dtype=np.float)
        self.node_counts_not_following_node = np.zeros(max_node_id+1, dtype=np.float)

    def process_variant(self, reference_node, variant_node):

        for node in (reference_node, variant_node):
            expected_count_following = 0
            expected_count_not_following = 0
            lookup = self.reverse_index.get_node_kmers_and_ref_positions(node)
            for result in zip(lookup[0], lookup[1]):
                kmer = result[0]
                ref_pos = result[1]
                kmer = int(kmer)
                nodes, ref_offsets, frequencies, allele_frequencies = self.kmer_index.get(kmer, max_hits=1000000)
                if nodes is None:
                    continue

                unique_ref_offsets, unique_indexes = np.unique(ref_offsets, return_index=True)

                if len(unique_indexes) == 0:
                    # Could happen when variant index has more kmers than full graph index
                    #logging.warning("Found not index hits for kmer %d" % kmer)
                    continue
                    
                n_hits = 0
                for index in unique_indexes:
                    # do not add count for the actual kmer we are searching for, we add 1 for this in the end
                    if ref_offsets[index] != ref_pos:
                        n_hits += allele_frequencies[index] # / frequencies[index]

                expected_count_following += n_hits
                expected_count_not_following += n_hits

            expected_count_following += 1

            self.node_counts_following_node[node] += expected_count_following
            self.node_counts_not_following_node[node] += expected_count_not_following

    def get_node_counts(self):
        for i, variant in enumerate(self.variants):
            if i % 1000 == 0:
                logging.info("%d reads processed" % i)

            try:
                reference_node, variant_node = self.graph.get_variant_nodes(variant)
            except VariantNotFoundException:
                continue

            self.process_variant(reference_node, variant_node)

        return self.node_counts_following_node, self.node_counts_not_following_node


class NodeCountModelCreatorFromSimpleChaining:
    def __init__(self, graph, reverse_kmer_index, nodes_followed_by_individual, individual_genome_sequence, kmer_index, n_nodes, n_reads_to_simulate=1000, read_length=150,  k=31, skip_chaining=False):
        self._graph = graph
        self._reverse_index = reverse_kmer_index
        self.kmer_index = kmer_index
        self.nodes_followed_by_individual = nodes_followed_by_individual
        self.genome_sequence = individual_genome_sequence
        self.reverse_genome_sequence = str(Seq(self.genome_sequence).reverse_complement())
        self._node_counts_following_node = np.zeros(n_nodes+1, dtype=np.float)
        self._node_counts_not_following_node = np.zeros(n_nodes+1, dtype=np.float)
        self.n_reads_to_simulate = n_reads_to_simulate
        self.read_length = read_length
        self.genome_size = len(self.genome_sequence)
        self._n_nodes = n_nodes
        self._k = k
        self._skip_chaining = skip_chaining

    def get_simulated_reads(self):
        reads = []
        for i in range(0, self.n_reads_to_simulate):
            if i % 100000 == 0:
                logging.info("%d/%d reads simulated" % (i, self.n_reads_to_simulate))
            pos_start = np.random.randint(0, self.genome_size - self.read_length)
            pos_end = pos_start + self.read_length

            for read in [self.genome_sequence[pos_start:pos_end], self.reverse_genome_sequence[pos_start:pos_end]]:
                reads.append(read)

        return reads

    def get_node_counts(self):
        # Simulate reads from the individual
        # for each read, find nodes in best chain
        # increase those node counts

        reads = self.get_simulated_reads()
        index = self.kmer_index

        chain_positions, node_counts = cython_chain_genotyper.run(reads, index._hashes_to_index,
              index._n_kmers,
              index._nodes,
              index._ref_offsets,
              index._kmers,
              index._frequencies,
              index._modulo,
              self._graph.node_to_edge_index,
              self._graph.edges,
              self._graph.node_to_n_edges,
              self._graph.ref_offset_to_node,
              self._reverse_index.nodes_to_index_positions,
              self._reverse_index.nodes_to_n_hashes,
              self._reverse_index.hashes,
              self._reverse_index.ref_positions,
              self._n_nodes,
              self._k,
              )

        #logging.info("Sum of positions: %d" % np.sum(chain_positions))
        #logging.info("Sum of node counts: %d" % np.sum(node_counts))
        array_nodes_followed_by_individual = np.zeros(self._n_nodes+1)
        array_nodes_followed_by_individual[self.nodes_followed_by_individual] = 1
        followed = np.where(array_nodes_followed_by_individual == 1)[0]
        #logging.info("N followd: %d nodes are followed by individual" % len(followed))
        not_followed = np.where(array_nodes_followed_by_individual == 0)[0]
        self._node_counts_following_node[followed] = node_counts[followed]
        self._node_counts_not_following_node[not_followed] = node_counts[not_followed]

        return self._node_counts_following_node, self._node_counts_not_following_node