import logging
import time

from pyfaidx import Fasta
from .genotyper import NodeCounts, ReadKmers
import pyximport; pyximport.install(language_level=3)
from graph_kmer_index.cython_kmer_index import get_nodes_and_ref_offsets_from_multiple_kmers as cython_index_lookup
from .chaining import chain, chain_with_score
import numpy as np
from Bio.Seq import Seq
#from graph_kmer_index import letter_sequence_to_numeric
from .letter_sequence_to_numeric import letter_sequence_to_numeric
from .count_genotyper import CountGenotyper
import math
from alignment_free_graph_genotyper import cython_chain_genotyper

def read_kmers(read, power_array):
    numeric = letter_sequence_to_numeric(read)
    return np.convolve(numeric, power_array, mode='valid')  # % 452930477


class NumpyNodeCounts:
    def __init__(self, node_counts):
        self.node_counts = node_counts

    def to_file(self, file_name):
        np.save(file_name, self.node_counts)

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name + ".npy")
        return cls(data)


class ChainGenotyper:
    def __init__(self, graph, sequence_graph, linear_path, reads, kmer_index, vcf_file_name, k,
                 truth_alignments=None, write_alignments_to_file=None, reference_k=7,
                 weight_chains_by_probabilities=False, max_node_id=None):

        self._max_node_id = max_node_id
        self._reads = reads
        self._graph = graph
        self._sequence_graph = sequence_graph
        self._linear_path = linear_path
        self._kmer_index = kmer_index
        self._vcf_file_name = vcf_file_name
        self._k = k
        self._reference_k = reference_k

        self._truth_alignments = truth_alignments
        self._node_counts = NodeCounts()
        self._has_correct_chain = []
        self._has_not_correct_chain = []
        self._best_chain_matches = []
        self._weight_chains_by_probabilities = weight_chains_by_probabilities
        self._reference_kmers = \
            ReadKmers.get_kmers_from_read_dynamic(str(Fasta("ref.fa")["1"]), np.power(4, np.arange(0, reference_k)))

        self._out_file_alignments = None
        if write_alignments_to_file is not None:
            logging.info("Will write positions of ailgnmlents to file: %s" % write_alignments_to_file)
            self._out_file_alignments = open(write_alignments_to_file, "w")

        self._power_array = np.power(4, np.arange(0, self._k))
        self._power_array_short = np.power(4, np.arange(0, self._reference_k))

        self._detected_chains = {}
        self.approx_read_length = 150

    @staticmethod
    def find_chains(ref_offsets, read_offsets, nodes, chain_position_threshold=2):
        # Sort everything
        potential_chain_start_positions = ref_offsets - read_offsets
        sorting = np.argsort(potential_chain_start_positions)
        ref_offsets = ref_offsets[sorting]
        nodes = nodes[sorting]
        potential_chain_start_positions = potential_chain_start_positions[sorting]
        read_offsets = read_offsets[sorting]

        # Find all potential chain starts (by grouping reference start positions that are close)
        chain_start_and_end_indexes = np.where(np.ediff1d(potential_chain_start_positions, to_begin=1000, to_end=1000)
                                               >= chain_position_threshold)[0]
        chains = []
        for start, end in zip(chain_start_and_end_indexes[0:-1], chain_start_and_end_indexes[1:]):
            # Score depends on number of unique read offsets that matches kmers that gives this start
            score = len(np.unique(read_offsets[start:end]))
            #logging.info("Score: %d" % score)

            chains.append([potential_chain_start_positions[start], nodes[start:end], score])

        return chains

    def _write_alignment_to_file(self, name, start_position, score, chromosome=1, nodes="",
                        chains="", counting_nodes="", best_chain_prob=0.0):
        if self._out_file_alignments is None:
            return
        self._out_file_alignments.writelines(["%d\t%d\t%d\t60\t%d\t,%s,\t%s\t%s\t%.4f\n"%
                                              (name, chromosome, start_position, score,
                                               ','.join((str(n) for n in nodes)),
                                               str([chain for chain in chains if chain[5] > 0]).replace("\n", ""),
                                                counting_nodes, best_chain_prob)])

    def _score_chains(self, chains, short_kmers):
        for chain in chains:
            ref_start = int(chain[0])
            ref_end = ref_start + self.approx_read_length
            reference_kmers = self._reference_kmers[ref_start:ref_end-self._reference_k]
            #logging.info("----- CHAIN %d" % chain[0])
            #logging.info("Ref kmers: %s" % reference_kmers)
            #logging.info("Short kmers: %s" % short_kmers)
            score = len(short_kmers.intersection(reference_kmers)) / len(set(short_kmers))
            chain[2] = score

    def _get_read_chains_only_one_direction(self, read):
        kmers = read_kmers(read, self._power_array)
        short_kmers = read_kmers(read, self._power_array_short)
        nodes, ref_offsets, read_offsets = self._kmer_index.get_nodes_and_ref_offsets_from_multiple_kmers(kmers)
        """
        nodes, ref_offsets, read_offsets = cython_index_lookup(
            kmers,
            self._kmer_index._hasher._hashes,
            self._kmer_index._hashes_to_index,
            self._kmer_index._n_kmers,
            self._kmer_index._nodes,
            self._kmer_index._ref_offsets
        )
        """

        if len(nodes) == 0:
            return []
        chains = ChainGenotyper.find_chains(ref_offsets, read_offsets, nodes)
        #chains = chain(ref_offsets, read_offsets, nodes)
        self._score_chains(chains, set(short_kmers))
        #chains = chain_with_score(ref_offsets, read_offsets, nodes, self._reference_kmers, short_kmers)
        return chains

    def _get_read_chains(self, read):
        # Gets all the chains for the read and its reverse complement
        chains = self._get_read_chains_only_one_direction(read)

        reverse_chains = self._get_read_chains_only_one_direction(str(Seq(read).reverse_complement()))
        chains.extend(reverse_chains)
        chains = sorted(chains, key=lambda c: c[2], reverse=True)
        return chains


    def get_counts(self):
        for i, read in enumerate(self._reads):
            if i % 1000 == 0:
                logging.info("%d reads processed" % i)

            chains = self._get_read_chains(read)
            if len(chains) == 0:
                logging.warning("Found no chains for %d" % i)
                continue
            best_chain = chains[0]
            #self._detected_chains[i] = chains

            for node in best_chain[1]:
                self._node_counts.add_count(node)

            self._check_correctness(i, chains, best_chain)

    def _check_correctness(self, i, chains, best_chain):
        if self._truth_alignments is not None:
            correct_ref_position = self._truth_alignments.positions[i]
            if abs(correct_ref_position - best_chain[0]) <= 150:
                self._best_chain_matches.append(i)
            else:
                pass
                #logging.info(" === Read %d, correct: %d ===" % (i, correct_ref_position))
                #logging.info(chains)

            correct_chain_found = False
            for chain in chains:
                if abs(correct_ref_position - chain[0]) <= 150:
                    correct_chain_found = True
                    break

            if correct_chain_found:
                self._has_correct_chain.append(i)

    def genotype(self):
        if self._truth_alignments is not None:
            logging.info("N correct chains found: %d" % len(self._has_correct_chain))
            logging.info("N best chain is correct: %d" % len(self._best_chain_matches))

        count_genotyper = CountGenotyper(self, self._graph, self._sequence_graph, self._vcf_file_name, self._linear_path)
        count_genotyper.genotype()

    def get_node_count(self, node):
        return math.floor(self._node_counts[node])


class CythonChainGenotyper(ChainGenotyper):
    def get_counts(self):
        start_time = time.time()
        index = self._kmer_index
        fasta_file_name = self._reads
        logging.info("Number of kmers in index: %d" % len(index._kmers))
        chain_positions, node_counts = cython_chain_genotyper.run(fasta_file_name,
                    index._hasher._hashes,
                    index._hashes_to_index,
                    index._n_kmers,
                    index._nodes,
                    index._ref_offsets,
                    index._kmers,
                    self._reference_kmers,
                    self._max_node_id
        )
        self._node_counts = NumpyNodeCounts(node_counts)

        end_time = time.time()
        logging.info("Time spent on getting node counts: %.5f" % (end_time - start_time))

    def genotype(self):
        count_genotyper = CountGenotyper(self, self._graph, self._sequence_graph, self._vcf_file_name, self._linear_path)
        count_genotyper.genotype()

    def get_node_count(self, node):
        return self._node_counts.node_counts[node]
