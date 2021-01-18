import logging
import time

from pyfaidx import Fasta
from graph_kmer_index import ReadKmers

from .genotyper import NodeCounts
import pyximport; pyximport.install(language_level=3)
#from graph_kmer_index.cython_kmer_index import get_nodes_and_ref_offsets_from_multiple_kmers as cython_index_lookup
from .chaining import chain, chain_with_score
import numpy as np
from Bio.Seq import Seq
#from graph_kmer_index import letter_sequence_to_numeric
from .letter_sequence_to_numeric import letter_sequence_to_numeric
from .count_genotyper import CountGenotyper
import math
from alignment_free_graph_genotyper import cython_chain_genotyper
from .genotyper import BestChainGenotyper

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
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(data)



class ChainGenotyper:
    def __init__(self, graph, sequence_graph, linear_path, reads, kmer_index, vcf_file_name, k,
                 truth_alignments=None, write_alignments_to_file=None, reference_k=7,
                 weight_chains_by_probabilities=False, max_node_id=None, reference_kmers=None,
                 unique_index=None, reverse_index=None, graph_edges=None, distance_to_node=None, skip_reference_kmers=False,
                 skip_chaining=False):

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
        if not skip_reference_kmers:
            if reference_kmers is None:
                self._reference_kmers = \
                    ReadKmers.get_kmers_from_read_dynamic(str(Fasta("ref.fa")["1"]), np.power(4, np.arange(0, reference_k)))
            else:
                self._reference_kmers = reference_kmers

        self._out_file_alignments = None
        if write_alignments_to_file is not None:
            logging.info("Will write positions of ailgnmlents to file: %s" % write_alignments_to_file)
            self._out_file_alignments = open(write_alignments_to_file, "w")

        self._power_array = np.power(4, np.arange(0, self._k))
        self._power_array_short = np.power(4, np.arange(0, self._reference_k))

        self._detected_chains = {}
        self.approx_read_length = 150
        self.unique_index = unique_index
        self.hits_against_unique_index = 0
        self._reverse_index = reverse_index
        self._graph_edges = graph_edges
        self._distance_to_node = distance_to_node
        self._skip_chaining = skip_chaining

    @staticmethod
    def find_chains(ref_offsets, read_offsets, nodes, frequencies, chain_position_threshold=2, kmers=None):
        # Sort everything
        potential_chain_start_positions = ref_offsets - read_offsets
        sorting = np.argsort(potential_chain_start_positions)
        frequencies = frequencies[sorting]
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
            #score = len(np.unique(read_offsets[start:end]))
            #logging.info("Score: %d" % score)
            unique_ref_offsets, indexes = np.unique(read_offsets[start:end], return_index=True)
            f = frequencies[start:end]
            score = np.sum(1 / f[indexes])
            #logging.info("Score: %.4f. %s" % (score, f[indexes]))

            chains.append([potential_chain_start_positions[start], nodes[start:end], score, kmers])

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
        nodes, ref_offsets, read_offsets, frequencies = self._kmer_index.get_nodes_and_ref_offsets_from_multiple_kmers(kmers)
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
        chains = ChainGenotyper.find_chains(ref_offsets, read_offsets, nodes, frequencies, kmers=kmers)
        #chains = chain(ref_offsets, read_offsets, nodes)
        self._score_chains(chains, set(short_kmers))
        #chains = chain_with_score(ref_offsets, read_offsets, nodes, self._reference_kmers, short_kmers)
        return chains

    def _get_read_chains(self, read):
        # Gets all the chains for the read and its reverse complement
        # First check match against unique index
        hit_against_unique = False
        if self.unique_index is not None:
            for direction in [-1, 1]:
                if direction == -1:
                    kmers = read_kmers(str(Seq(read).reverse_complement()), self._power_array)
                else:
                    kmers = read_kmers(read, self._power_array)

                for kmer_pos, kmer in enumerate(kmers):
                    nodes = set(self.unique_index.get(kmer))
                    if len(nodes) > 0:
                        hit_against_unique = True
                        self.hits_against_unique_index += 1
                        for node in nodes:
                            self._node_counts.add_count(node)

        if hit_against_unique:
            return None

        chains = self._get_read_chains_only_one_direction(read)
        reverse_chains = self._get_read_chains_only_one_direction(str(Seq(read).reverse_complement()))
        chains.extend(reverse_chains)
        chains = sorted(chains, key=lambda c: c[2], reverse=True)
        return chains


    def get_counts(self):
        for i, read in enumerate(self._reads):
            if i % 1000 == 0:
                logging.info("%d reads processed. Hits against unique index: %d" % (i, self.hits_against_unique_index))

            chains = self._get_read_chains(read)
            if chains is None:
                continue

            if len(chains) == 0:
                logging.warning("Found no chains for %d" % i)
                continue
            best_chain = chains[0]
            #self._detected_chains[i] = chains

            if self._reverse_index is not None:
                matching_nodes = BestChainGenotyper.align_nodes_to_read(self._reverse_index, self._graph, self._linear_path, int(best_chain[0]), best_chain[3], 150, set(best_chain[1]))
                #logging.info("Matching nodes with reverse: %s. Original nodes: %s" % (matching_nodes, best_chain[1]))
                for node in matching_nodes:
                    self._node_counts.add_count(node)
            else:
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
                    index._hashes_to_index,
                    index._n_kmers,
                    index._nodes,
                    index._ref_offsets,
                    index._kmers,
                    index._frequencies,
                    index._modulo,
                    self._max_node_id,
                    self._k,
                    self._skip_chaining

        )
        self.chain_positions = chain_positions
        self._node_counts = NumpyNodeCounts(node_counts)

        end_time = time.time()
        logging.info("Time spent on getting node counts: %.5f" % (end_time - start_time))

    def genotype(self):
        count_genotyper = CountGenotyper(self, self._graph, self._sequence_graph, self._vcf_file_name, self._linear_path)
        count_genotyper.genotype()

    def get_node_count(self, node):
        return self._node_counts.node_counts[node]


class UniqueKmerGenotyper:
    def __init__(self, unique_index, reads, k, max_node=13000000):
        self._unique_index = unique_index
        self._reads = reads
        self._k = k
        self._power_array = np.power(4, np.arange(0, self._k))
        self._node_counts = NumpyNodeCounts(np.zeros(max_node))

    def get_counts(self):
        n_reads_hit = 0
        logging.info("Getting counts for %d reads" % len(self._reads))
        prev_time = time.time()
        i = 0
        debug_kmers = [ 327561780834057096, 3785481886904335240,  615792156985768840, 614947749235506056,  327561798013926280, 4074556687986178952,
            326717355903925128,  615792174165638024, 4073712245876177800,
           3786326311834467208,  614947732055636872, 4073712263056046984,
           4074556670806309768, 3786326294654598024,  326717373083794312,
           3785481869724466056]

        debug_kmers = [94443029691480101, 94495806249613349]
        for read in self._reads:
            if read.startswith(">"):
                continue

            if i % 1000 == 0:
                logging.info("%d lines processed. N reads hit index: %d. Time on last 1000: %.4f" % (i, n_reads_hit, time.time() - prev_time))
                prev_time = time.time()

            for direction in [-1, 1]:

                if direction == -1:
                    kmers = read_kmers(str(Seq(read).reverse_complement()), self._power_array)
                else:
                    kmers = read_kmers(read, self._power_array)

                for kmer_pos, kmer in enumerate(kmers):
                    if kmer in debug_kmers:
                        logging.info("HIT! Read %d, kmer pos: %d, dir: %d. Kmer: %d. Read: %s" % (i, kmer_pos, direction, kmer, read))

                    nodes = set(self._unique_index.get(kmer))
                    if len(nodes) > 0:
                        n_reads_hit += 1
                        for node in nodes:
                            self._node_counts.node_counts[node] += 1
                        #break
            i += 1



