import logging
from pyfaidx import Fasta
from graph_kmer_index import ReadKmers

from .genotyper import NodeCounts
import pyximport; pyximport.install(language_level=3)
from .chaining import chain, chain_with_score

import numpy as np
from Bio.Seq import Seq

from .count_genotyper import CountGenotyper
import math
from .letter_sequence_to_numeric import letter_sequence_to_numeric


def get_power_array(k):
    return np.power(4, np.arange(0, k))


def read_kmers(read, power_array=None):
    numeric = letter_sequence_to_numeric(read)
    return np.convolve(numeric, power_array, mode='valid')  # % 452930477


class ChainMapper:
    def __init__(self, graph, reads, kmer_index, reverse_kmer_index, k, max_node_id=None, max_reads=1000000, linear_reference_kmers=None):

        self._graph = graph
        self._reads = reads
        self._kmer_index = kmer_index
        self._reverse_index = reverse_kmer_index
        self._k = k
        self._max_node_id = max_node_id
        self._reference_kmers = linear_reference_kmers

        self._node_counts = np.zeros(max_node_id+1)
        self._positions = np.zeros(max_reads)
        self._power_array = np.power(4, np.arange(0, self._k))

        self.approx_read_length = 150
        self._n_variant_node_unique_increased = 0
        self._n_ref_node_unique_increased = 0
        self._n_variant_node_has_duplicate = 0
        self._n_ref_node_has_duplicate = 0
        self._n_ambiguous = 0

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
            score = len(np.unique(read_offsets[start:end]))
            #logging.info("Score: %d" % score)
            #unique_ref_offsets, indexes = np.unique(read_offsets[start:end], return_index=True)
            #f = frequencies[start:end]
            #score = np.sum(1 / f[indexes])
            #logging.info("Score: %.4f. %s" % (score, f[indexes]))

            chains.append([potential_chain_start_positions[start], nodes[start:end], score, kmers])

        return chains

    def _get_read_chains_only_one_direction(self, read):
        kmers = read_kmers(read, self._power_array)
        nodes, ref_offsets, read_offsets, frequencies = self._kmer_index.get_nodes_and_ref_offsets_from_multiple_kmers(kmers, max_hits=100)

        if len(nodes) == 0:
            return []
        chains = ChainMapper.find_chains(ref_offsets, read_offsets, nodes, frequencies, kmers=kmers)
        return chains

    def _get_read_chains(self, read):
        chains = self._get_read_chains_only_one_direction(read)
        reverse_chains = self._get_read_chains_only_one_direction(str(Seq(read).reverse_complement()))
        chains.extend(reverse_chains)
        chains = sorted(chains, key=lambda c: c[2], reverse=True)
        return chains

    def increase_node_counts_for_chain(self, chain):
        pass

    def align_nodes_to_read(self, read_ref_pos, read_kmers, read_length=150, read_sequence=None):
        read_kmers = set(read_kmers)
        # Find all pairs of variant nodes in this area of graph
        # if only one node has match in read_kmers, increase that node, done
        # if both nodes have match in read_kmers:
        #   if variant has duplicate on linear ref somewhere, choose ref node
        #   else choose variant node

        # More specifically:
        # if none of the variant kmers match and any of the ref kmers match: increase ref node
        # if none of the ref kmers match and any of the variant kmers match: increase variant node
        # if 1 or more of variant kmers match and 1 or more of ref kmers match:
        #   if any of the matching variant kmers also exist on linear ref somewhere nearby, and none of the matching ref kmers exist on linear ref somewhere else:
        #       increase ref
        #   elif any of ref kmers also exist on ref somewhere nearby (not on same position as themself) and not case for any variant kmers:
        #       increase variant
        #   else:
        #       it's a tie, shouldn't happen frequently, maybe increase both with 0.5?
        #

        variant_nodes = self._graph.get_variant_nodes_in_region(1, read_ref_pos, read_ref_pos + 150)
        for variant_node_pair in variant_nodes:
            assert len(variant_node_pair) == 2, "Only biallelic supported now. Nodes: %s" % variant_node_pair
            ref_node = variant_node_pair[0]
            variant_node = variant_node_pair[1]
            if variant_node in self._graph.linear_ref_nodes() or self._graph.get_node_size(variant_node) == 0:
                ref_node = variant_node_pair[1]
                variant_node = variant_node_pair[0]

            variant_kmers_lookup = self._reverse_index.get_node_kmers_and_ref_positions(variant_node)
            variant_kmers = list(zip(variant_kmers_lookup[0], variant_kmers_lookup[1]))
            variant_kmers_found_in_read = [kmer for kmer, ref_pos in variant_kmers if kmer in read_kmers]

            ref_kmers_lookup = self._reverse_index.get_node_kmers_and_ref_positions(ref_node)
            ref_kmers = list(zip(ref_kmers_lookup[0], ref_kmers_lookup[1]))
            ref_kmers_found_in_read = [kmer for kmer, ref_pos in ref_kmers if kmer in read_kmers]

            increase_variant_node = False
            increase_ref_node = False
            if len(variant_kmers_found_in_read) > 0 and len(ref_kmers_found_in_read) == 0:
                increase_variant_node = True
                self._n_variant_node_unique_increased += 1
            elif len(variant_kmers_found_in_read) == 0 and len(ref_kmers_found_in_read) > 0:
                increase_ref_node = True
                self._n_ref_node_unique_increased += 1
            elif len(variant_kmers_found_in_read) > 0 and len(ref_kmers_found_in_read) > 0:
                ref_area_start = read_ref_pos
                ref_area_end = ref_area_start + read_length
                variant_kmers_also_existing_nearby = [(kmer, ref_pos) for kmer, ref_pos in variant_kmers if kmer in variant_kmers_found_in_read and kmer in self._reference_kmers.get_between_except(ref_area_start, ref_area_end, ref_pos)]
                ref_kmers_also_existing_nearby = [(kmer, ref_pos) for kmer, ref_pos in ref_kmers if kmer in ref_kmers_found_in_read and kmer in self._reference_kmers.get_between_except(ref_area_start, ref_area_end, ref_pos)]

                if len(variant_kmers_also_existing_nearby) > 0 and len(ref_kmers_also_existing_nearby) == 0:
                    increase_ref_node = True
                    self._n_variant_node_has_duplicate += 1
                elif len(variant_kmers_also_existing_nearby) == 0 and len(ref_kmers_also_existing_nearby) > 0:
                    increase_variant_node = True
                    self._n_ref_node_has_duplicate += 1
                else:
                    logging.warning("Ambigous case when doing node lookup: Both variant and ref node have kmers existing on linear ref elsewhere. Nodes: %d/%d" % (ref_node, variant_node))
                    logging.warning("Looking up linear ref kmers between %d and %d" % (ref_area_start, ref_area_end))
                    logging.warning("Variant kmers found in read:               %s" % variant_kmers_found_in_read)
                    logging.warning("All variant kmers          :               %s" % variant_kmers)
                    logging.warning("Variant kmers also existing on linear ref: %s" % variant_kmers_also_existing_nearby)
                    logging.warning("Ref     kmers found in read:               %s" % ref_kmers_found_in_read)
                    logging.warning("Ref  kmers also existing on linear ref:    %s" % ref_kmers_also_existing_nearby)
                    logging.warning("All ref     kmers          :               %s" % ref_kmers)
                    self._n_ambiguous += 1

            if increase_variant_node:
                if variant_node == 176732:
                    logging.warning("")
                    logging.warning("")
                    logging.warning("INCREASING VARIANT NODE %d on read with seq %s" % (variant_node, read_sequence))
                    logging.warning("")
                    logging.warning("")
                self._node_counts[variant_node] += 1
            elif increase_ref_node:
                self._node_counts[ref_node] += 1

    def get_counts(self):
        for i, read in enumerate(self._reads):
            if i % 1000 == 0:
                logging.info("%d reads processed." % i)
                logging.info("N trivial variant nodes: %d. N trivial ref nodes: %d. N variant nodes with duplicate kmer: %d. N ref nodes with duplicate kmers: %d. N ambiguous: %d" % (self._n_variant_node_unique_increased, self._n_ref_node_unique_increased, self._n_variant_node_has_duplicate, self._n_ref_node_has_duplicate, self._n_ambiguous))

            chains = self._get_read_chains(read)
            if chains is None:
                continue

            if len(chains) == 0:
                logging.warning("Found no chains for %d" % i)
                continue
            best_chain = chains[0]
            self._positions[i] = int(best_chain[0])
            #self.align_nodes_to_read(int(best_chain[0]), best_chain[3], 150, read)
            self._node_counts[best_chain[1]] += 1

        return self._node_counts, self._positions


