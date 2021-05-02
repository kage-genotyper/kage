import numpy as np
from Bio.Seq import Seq
import logging
from obgraph import Graph, VariantNotFoundException
from graph_kmer_index import ReverseKmerIndex, KmerIndex
import time
from alignment_free_graph_genotyper import cython_chain_genotyper

class GenotypeNodeCountModel:
    def __init__(self, counts_homo_ref, counts_homo_alt, counts_hetero):
        self.counts_homo_ref = counts_homo_ref
        self.counts_homo_alt = counts_homo_alt
        self.counts_hetero = counts_hetero

    @classmethod
    def from_node_count_model(cls, model, variant_nodes):
        ref_nodes = variant_nodes.ref_nodes
        var_nodes = variant_nodes.var_nodes

        n = len(model.node_counts_following_node)
        counts_homo_ref = np.zeros(n)
        counts_homo_alt = np.zeros(n)
        counts_hetero = np.zeros(n)

        counts_homo_ref[ref_nodes] = model.node_counts_following_node[ref_nodes] * 2
        counts_homo_ref[var_nodes] = model.node_counts_not_following_node[var_nodes] * 2

        counts_homo_alt[var_nodes] = model.node_counts_following_node[var_nodes] * 2
        counts_homo_alt[ref_nodes] = model.node_counts_not_following_node[ref_nodes] * 2

        counts_hetero[var_nodes] = model.node_counts_following_node[var_nodes] + model.node_counts_not_following_node[var_nodes]
        counts_hetero[ref_nodes] = model.node_counts_following_node[ref_nodes] + model.node_counts_not_following_node[ref_nodes]

        return cls(counts_homo_ref, counts_homo_alt, counts_hetero)

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(data["counts_homo_ref"], data["counts_homo_alt"], data["counts_hetero"])

    def to_file(self, file_name):
        np.savez(file_name, counts_homo_ref=self.counts_homo_ref, counts_homo_alt=self.counts_homo_alt,
                 counts_hetero=self.counts_hetero)


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
    def __init__(self, kmer_index: KmerIndex, reverse_index: ReverseKmerIndex, variant_to_nodes, variant_start_id, variant_end_id, max_node_id,
                 scale_by_frequency=False, allele_frequency_index=None):
        self.kmer_index = kmer_index
        self.variant_to_nodes = variant_to_nodes
        self.reverse_index = reverse_index
        self.variant_start_id = variant_start_id
        self.variant_end_id = variant_end_id

        self.node_counts_following_node = np.zeros(max_node_id+1, dtype=np.float)
        self.node_counts_not_following_node = np.zeros(max_node_id+1, dtype=np.float)
        self._scale_by_frequency = scale_by_frequency
        self._allele_frequency_index = allele_frequency_index
        if self._allele_frequency_index is not None:
            logging.info("Will fetch allele frequencies from allele frequency index")

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
                    if self._allele_frequency_index is None:
                        allele_frequency = allele_frequencies[index]  # fetch from graph
                    else:
                        # get the nodes belonging to this ref offset
                        ref_offset = ref_offsets[index]
                        hit_nodes = nodes[np.where(ref_offsets == ref_offset)]
                        # allele frequency is the lowest allele frequency for these nodes
                        allele_frequency = np.min(self._allele_frequency_index[hit_nodes])

                    if ref_offsets[index] != ref_pos:

                        if self._scale_by_frequency:
                            n_hits += allele_frequency / frequencies[index]
                        else:
                            n_hits += allele_frequency

                expected_count_following += n_hits
                expected_count_not_following += n_hits

                if self._scale_by_frequency and False:
                    # We add counts for following node here
                    for hit_node, ref_offset, frequency, allele_frequency in zip(nodes, ref_offsets, frequencies, allele_frequencies):
                        if hit_node == node and ref_offset == ref_pos:
                            expected_count_following += allele_frequency * 1 / frequency

            if not self._scale_by_frequency or True:
                expected_count_following += 1.0


            self.node_counts_following_node[node] += expected_count_following
            self.node_counts_not_following_node[node] += expected_count_not_following

    def get_node_counts(self):
        for i, variant_id in enumerate(range(self.variant_start_id, self.variant_end_id)):
            if i % 25000 == 0:
                logging.info("%d/%d variants processed" % (i, self.variant_end_id-self.variant_start_id))

            #reference_node, variant_node = self.graph.get_variant_nodes(variant)
            reference_node = self.variant_to_nodes.ref_nodes[variant_id]
            variant_node = self.variant_to_nodes.var_nodes[variant_id]

            if reference_node == 0 or variant_node == 0:
                continue

            self.process_variant(reference_node, variant_node)

        return self.node_counts_following_node, self.node_counts_not_following_node


class NodeCountModelCreatorFromSimpleChaining:
    def __init__(self, graph, reference_index, nodes_followed_by_individual, individual_genome_sequence, kmer_index, n_nodes, n_reads_to_simulate=1000, read_length=150,  k=31, skip_chaining=False, max_index_lookup_frequency=5, reference_index_scoring=None, seed=None):
        self._graph = graph
        self._reference_index = reference_index
        self.kmer_index = kmer_index
        self.nodes_followed_by_individual = nodes_followed_by_individual
        self.genome_sequence = individual_genome_sequence
        #self.reverse_genome_sequence = str(Seq(self.genome_sequence).reverse_complement())
        self._node_counts_following_node = np.zeros(n_nodes+1, dtype=np.float)
        self._node_counts_not_following_node = np.zeros(n_nodes+1, dtype=np.float)
        self.n_reads_to_simulate = n_reads_to_simulate
        self.read_length = read_length
        self.genome_size = len(self.genome_sequence)
        self._n_nodes = n_nodes
        self._k = k
        self._skip_chaining = skip_chaining
        self._max_index_lookup_frequency = max_index_lookup_frequency
        self._reference_index_scoring = reference_index_scoring
        self._seed = seed
        if self._seed is None:
            self._seed = np.random.randint(0, 1000)

    def get_simulated_reads(self):
        np.random.seed(self._seed)
        reads = []
        prev_time = time.time()
        read_positions_debug = []
        for i in range(0, self.n_reads_to_simulate):
            if i % 500000 == 0:
                logging.info("%d/%d reads simulated (time spent on chunk: %.3f)" % (i, self.n_reads_to_simulate, time.time()-prev_time))
                prev_time = time.time()


            pos_start = np.random.randint(0, self.genome_size - self.read_length)
            pos_end = pos_start + self.read_length

            if i < 20:
                read_positions_debug.append(pos_start)

            reads.append(self.genome_sequence[pos_start:pos_end])
            # Don't actually need to simulate from reverse complement, mapper is anyway reversecomplementing every sequence
            #for read in [self.genome_sequence[pos_start:pos_end], self.reverse_genome_sequence[pos_start:pos_end]]:
            #for read in [self.genome_sequence[pos_start:pos_end]]:
                #yield read
            #    reads.append(read)

        logging.info("First 20 read positions: %s" % read_positions_debug)

        return reads

    def get_node_counts(self):
        # Simulate reads from the individual
        # for each read, find nodes in best chain
        # increase those node counts

        reads = self.get_simulated_reads()
        # Set to none to not use memory on the sequence anymore
        self.genome_sequence = None

        logging.info("Getting node counts")
        chain_positions, node_counts = cython_chain_genotyper.run(reads, self.kmer_index,
              self._n_nodes,
              self._k,
              self._reference_index,
              self._max_index_lookup_frequency,
              True,
              self._reference_index_scoring,
              self._skip_chaining
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