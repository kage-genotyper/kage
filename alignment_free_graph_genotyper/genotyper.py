import logging
from graph_kmer_index.kmer_index import KmerIndex
from graph_kmer_index import kmer_to_hash_fast
from graph_kmer_index import letter_sequence_to_numeric
from collections import defaultdict
from .count_genotyper import CountGenotyper
import math
import numpy as np
from Bio.Seq import Seq
import itertools

CHAR_VALUES = {"a": 0, "g": 1, "c": 2, "t": 3, "n": 0, "A": 0, "G": 1, "C": 2, "T": 3, "N": 0}

class ReadKmers:
    def __init__(self, kmers):
        self.kmers = kmers
        self._power_vector = None

    @classmethod
    def from_fasta_file(cls, fasta_file_name, k):

        power_vector = np.power(4, np.arange(0, k))
        f = open(fasta_file_name)
        kmers = itertools.chain(
            (ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector)
                    for line in f if not line.startswith(">")),
            (ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector)
                    for line in f if not line.startswith(">"))
        )

        return cls(kmers)

    @classmethod
    def from_list_of_string_kmers(cls, string_kmers):
        kmers = [
            [kmer_to_hash_fast(letter_sequence_to_numeric(k), len(k)) for k in read_kmers]
            for read_kmers in string_kmers
        ]
        return cls(kmers)

    @staticmethod
    def get_kmers_from_read(read, k):
        kmers = []
        for i in range(len(read) - k):
            letter_sequence = letter_sequence_to_numeric(read[i:i+k])
            kmers.append(kmer_to_hash_fast(letter_sequence, k))
        return kmers

    @staticmethod
    def get_kmers_from_read_dynamic(read, power_vector):
        #a = np.power(4, np.arange(0, k))
        #print(a)
        read = letter_sequence_to_numeric(read)
        #print(read)
        #print(np.convolve(read, a, mode='valid'))
        return np.convolve(read, power_vector, mode='valid')

    @staticmethod
    def get_kmers_from_read_dynamic_slow(read, k):
        read = letter_sequence_to_numeric(read)
        kmers = np.zeros(len(read)-k+1, dtype=np.int64)
        current_hash = kmer_to_hash_fast(read[0:k], k)
        kmers[0] = current_hash
        for i in range(1, len(read)-k+1):
            kmers[i] = (kmers[i-1] - np.power(4, k-1) * read[i-1]) * 4 + read[i+k-1]
            #assert kmers[i] == kmer_to_hash_fast(read[i:i+k], k), "New hash %d != correct %d" % (kmers[i], kmer_to_hash_fast(read[i:i+k], k))

        return kmers

    def __iter__(self):
        return self.kmers.__iter__()

    def __next__(self):
        return self.kmers.__next__()




class BaseGenotyper:
    def __init__(self, graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k):
        self._graph = graph
        self._vcf_file_name = vcf_file_name
        self._sequence_graph = sequence_graph
        self._linear_path = linear_path
        self._reference_nodes = linear_path.nodes_in_interval()
        self._kmer_index = kmer_index
        self._read_kmers = read_kmers
        self._k = k


class IndependentKmerGenotyper(BaseGenotyper):
    def __init__(self, graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k):
        super().__init__(graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k)

        self._node_counts = defaultdict(float)

    def genotype(self):
        self.get_counts()
        count_genotyper = CountGenotyper(self, self._graph, self._sequence_graph, self._vcf_file_name, self._linear_path)
        count_genotyper.genotype()

    def get_node_count(self, node):
        return math.ceil(self._node_counts[node])

    def get_counts(self):
        for i, read_kmers in enumerate(self._read_kmers):
            if i % 1000 == 0:
                logging.info("%d reads procesed" % i)
            for hash in read_kmers:
                nodes = self._kmer_index.get(hash)
                if nodes is not None:
                    for node in nodes:
                        self._node_counts[node] += 1 / len(nodes)




class BestChainGenotyper(BaseGenotyper):
    def __init__(self, graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k, truth_alignments=None):
        super().__init__(graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k)

        self._node_counts = defaultdict(float)

    @staticmethod
    def get_nodes_in_best_chain(nodes, ref_offsets, expected_read_length=150, min_chaining_score=2):
        # Returns nodes in best chain. Ref-offsets may be non-unique (since there are multiple nodes for each kmer.
        # Each node comes with one ref offset)
        sorting = np.argsort(ref_offsets)
        nodes = nodes[sorting]
        ref_offsets = ref_offsets[sorting]
        chain_end_indexes = np.where(np.ediff1d(ref_offsets, to_end=2**30) >= expected_read_length)[0] + 1
        best_start = None
        best_end = None
        best_chain_score = 0
        current_chain_start_index = 0
        for chain_end in chain_end_indexes:
            offsets = ref_offsets[current_chain_start_index:chain_end]
            score = len(np.unique(offsets))
            #print("Checking chain start/end: %d/%d. Offsets: %s. Score: %d" % (current_chain_start_index, chain_end, offsets, score))
            if score > best_chain_score:
                #print("  New best")
                best_start = current_chain_start_index
                best_end = chain_end
                best_chain_score = score
            current_chain_start_index = chain_end

        if best_chain_score < min_chaining_score:
            return []
        else:
            return nodes[best_start:best_end]

    def _find_best_chains(self):
        for i, read_kmers in enumerate(self._read_kmers):
            if i % 1000 == 0:
                logging.info("%d reads processed (best chain genotyper" % i)
            all_nodes = []
            all_ref_offsets = []
            for hash in read_kmers:
                nodes, ref_offsets = self._kmer_index.get_nodes_and_ref_offsets(hash)
                if nodes is None:
                    continue
                all_nodes.append(nodes)
                all_ref_offsets.append(ref_offsets)

            if len(all_nodes) == 0:
                continue

            all_nodes = np.concatenate(all_nodes)
            all_ref_offsets = np.concatenate(all_ref_offsets)

            best_chain_nodes = BestChainGenotyper.get_nodes_in_best_chain(all_nodes, all_ref_offsets)

            for node in best_chain_nodes:
                self._node_counts[node] += 1

    def genotype(self):
        self._find_best_chains()
        count_genotyper = CountGenotyper(self, self._graph, self._sequence_graph, self._vcf_file_name, self._linear_path)
        count_genotyper.genotype()

    def get_node_count(self, node):
        return math.ceil(self._node_counts[node])



