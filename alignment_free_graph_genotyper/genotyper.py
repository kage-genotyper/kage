import logging
from graph_kmer_index.kmer_index import KmerIndex
from graph_kmer_index import kmer_to_hash_fast
#from graph_kmer_index import letter_sequence_to_numeric
from .letter_sequence_to_numeric import letter_sequence_to_numeric
from collections import defaultdict
from .count_genotyper import CountGenotyper
import math
import numpy as np
from Bio.Seq import Seq
import itertools
import sys
from pyfaidx import Fasta
import pickle

CHAR_VALUES = {"a": 0, "g": 1, "c": 2, "t": 3, "n": 0, "A": 0, "G": 1, "C": 2, "T": 3, "N": 0}

class ReadKmers:
    def __init__(self, kmers):
        self.kmers = kmers
        self._power_vector = None

    @classmethod
    def from_fasta_file(cls, fasta_file_name, k, small_k=None, smallest_k=8):
        power_vector = np.power(4, np.arange(0, k))
        f = open(fasta_file_name)
        f = [l for l in f.readlines() if not l.startswith(">")]
        logging.info("Number of lines: %d" % len(f))
        if small_k is None:
            kmers = itertools.chain(
                (ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector)
                        for line in f if not line.startswith(">")),
                (ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector)
                        for line in f if not line.startswith(">"))
            )
        else:
            power_vector_small = np.power(4, np.arange(0, small_k))
            power_vector_smallest = np.power(4, np.arange(0, smallest_k))
            kmers = zip(
                    (itertools.chain(

                        ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector),
                        ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector)

                    ) for line in f),
                    (itertools.chain(
                            ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector_small),
                            ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector_small)
                    )
                    for line in f),
                    (itertools.chain(
                        ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector_smallest),
                        ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector_smallest)
                    )
                    for line in f)
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


class NodeCounts:
    def __init__(self):
        self._node_counts = defaultdict(float)

    def to_file(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self._node_counts, f)

    @classmethod
    def from_file(cls, file_name):
        f = open(file_name, "rb")
        counts = pickle.load(f)
        object = cls()
        object._node_counts = counts
        return object

    def add_count(self, node):
        self._node_counts[node] += 1

    def __setitem__(self, key, value):
        self._node_counts[key] = value

    def __getitem__(self, item):
        return self._node_counts[item]


class BaseGenotyper:
    def __init__(self, graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k, reference_k=7):
        self._graph = graph
        self._vcf_file_name = vcf_file_name
        self._sequence_graph = sequence_graph
        self._linear_path = linear_path
        self._reference_nodes = linear_path.nodes_in_interval()
        self._kmer_index = kmer_index
        self._read_kmers = read_kmers
        self._reference_k = reference_k
        self._k = k
        logging.info("Creating reference kmers with kmersize %d" % reference_k)
        self._reference_kmers = ReadKmers.get_kmers_from_read_dynamic(str(Fasta("ref.fa")["1"]), np.power(4, np.arange(0, reference_k)))
        logging.info("Done creating reference kmers")



class IndependentKmerGenotyper(BaseGenotyper):
    def __init__(self, graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k):
        super().__init__(graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k)

        self._node_counts = NodeCounts()

    def genotype(self):
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
    def __init__(self, graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k,
                 truth_alignments=None, write_alignments_to_file=None, reference_k=7, weight_chains_by_probabilities=False,
                 remap_best_chain_to=None, align_nodes_to_reads=None):
        super().__init__(graph, sequence_graph, linear_path, read_kmers, kmer_index, vcf_file_name, k, reference_k=reference_k)

        self._truth_alignments = truth_alignments
        self._node_counts = NodeCounts()
        self._has_correct_chain = []
        self._has_not_correct_chain = []
        self._best_chain_matches = []
        self._weight_chains_by_probabilities = weight_chains_by_probabilities
        self._remap_best_chain_to = remap_best_chain_to

        if self._weight_chains_by_probabilities:
            logging.info("Chains will by weighted by probability of being correct chain")

        self._align_nodes_to_reads = align_nodes_to_reads

        self._out_file_alignments = None
        if write_alignments_to_file is not None:
            logging.info("Will write positions of ailgnmlents to file: %s" % write_alignments_to_file)
            self._out_file_alignments = open(write_alignments_to_file, "w")

    def write_alignment(self, name, start_position, score, chromosome=1, nodes="", chains="", counting_nodes="", best_chain_prob=0.0):
        if self._out_file_alignments is None:
            return
        self._out_file_alignments.writelines(["%d\t%d\t%d\t60\t%d\t,%s,\t%s\t%s\t%.4f\n"%  (name, chromosome, start_position,
                                                                                      score,
                                                                                      ','.join((str(n) for n in nodes)), str([chain for chain in chains if chain[5] > 0]).replace("\n", ""),
                                                                                      counting_nodes, best_chain_prob)])

    @staticmethod
    def get_nodes_in_best_chain(nodes, ref_offsets, read_offsets=None, expected_read_length=150, min_chaining_score=0):
        # Returns nodes in best chain. Ref-offsets may be non-unique (since there are multiple nodes for each kmer.
        # Each node comes with one ref offset)
        sorting = np.argsort(ref_offsets)
        nodes = nodes[sorting]
        ref_offsets = ref_offsets[sorting]
        if read_offsets is not None:
            read_offsets = read_offsets[sorting]
        chain_end_indexes = np.where(np.ediff1d(ref_offsets, to_end=2**30) >= expected_read_length)[0] + 1
        #print(chain_end_indexes)
        best_start = None
        best_end = None
        best_chain_score = 0
        current_chain_start_index = 0
        for chain_end in chain_end_indexes:
            offsets = ref_offsets[current_chain_start_index:chain_end]
            score = len(np.unique(offsets))
            #score = np.max(offsets) - np.min(offsets)
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
            if read_offsets is not None:
                return nodes[best_start:best_end], read_offsets[best_start:best_end]
            return nodes[best_start:best_end]

    @staticmethod
    def get_all_chains_with_scores(nodes, ref_offsets, read_offsets, reference_kmers, short_read_kmers, expected_read_length=150, min_chaining_score=0, short_k=7):
        sorting = np.argsort(ref_offsets)
        nodes = nodes[sorting]
        ref_offsets = ref_offsets[sorting]
        read_offsets = read_offsets[sorting]
        chain_end_indexes = np.where((np.ediff1d(ref_offsets, to_end=2**30) >= expected_read_length) |
                                     (np.ediff1d(read_offsets, to_end=2**30) < 0) |
                                     (np.abs(np.ediff1d(read_offsets, to_end=2**30) - np.ediff1d(ref_offsets, to_end=2**30)) > 3)  # If read offset increases with 3 more than ref offset increases with, means there is an insertion of more than 3 bp in read. abs checks both ways
                                     )[0] + 1
        #print(chain_end_indexes)
        current_chain_start_index = 0
        detected_chains = []
        #logging.info("============")
        #logging.info("NODES:          %s" % nodes)
        #logging.info("Ref offsets:    %s" % ref_offsets)
        #logging.info("Read offsets:   %s" % read_offsets)
        for chain_end in chain_end_indexes:
            offsets = ref_offsets[current_chain_start_index:chain_end]
            #score = len(np.unique(offsets))

            start_index = current_chain_start_index
            reference_start = int(ref_offsets[start_index] - read_offsets[start_index])
            reference_end = int(ref_offsets[chain_end-1] + (expected_read_length - read_offsets[chain_end-1]))

            #logging.info("--")
            #logging.info("Chain start index: %d" % start_index)
            #logging.info("Reference start: %d" % reference_start)
            #logging.info("Reference end: %d" % reference_end)
            #logging.info("Chain start ref: %d" % ref_offsets[start_index])
            #logging.info("Chain start in read: %d" % read_offsets[start_index])
            local_ref_kmers = reference_kmers[reference_start:reference_end - short_k]
            #logging.info("Chosen nodes: %s" % nodes[start_index:chain_end])
            #logging.info("Local ref kmers:  %s" % local_ref_kmers)
            #logging.info("Short read kmers: %s" % short_read_kmers)
            score = len(short_read_kmers.intersection(set(local_ref_kmers))) / len(short_read_kmers)
            #logging.info(local_ref_kmers[0:20])
            #logging.info(short_read_kmers[0:20])
            #score = len([i for i, read_kmer in enumerate(short_read_kmers[0:150-short_k])
            #                 if read_kmer in local_ref_kmers[i-1:i+2]
            #           ]) / (150-short_k)


            #logging.info("Ref start %d, SCORE: %.5f" % (reference_start, score))

            #if reference_start == 612041:
            #sys.exit()


            #score = np.max(offsets) - np.min(offsets)
            #print("Checking chain start/end: %d/%d. Offsets: %s. Score: %d" % (current_chain_start_index, chain_end, offsets, score))

            detected_chains.append([nodes[start_index:chain_end], current_chain_start_index, chain_end, score, reference_start, 0])
            current_chain_start_index = chain_end

        return sorted(detected_chains, key=lambda d: d[3], reverse=True)


    @staticmethod
    def assign_probabilities_to_chains(chains):
        # If only 1 chain, set prob to 1
        if len(chains) == 1:
            chains[0][5] = 1.0
            return

        best_chain_score = chains[0][3]
        n_good_chains = 1
        ref_positions_already_having_chain = set([chains[0][4]])
        for i, chain in enumerate(chains[1:]):
            if chain[3] >= best_chain_score * 0.98 and chain[4] not in ref_positions_already_having_chain:
                #logging.info("   Good chain: %s" % chain)
                n_good_chains += 1
                ref_positions_already_having_chain.add(chain[4])

        #logging.info("N good chains: %d" % n_good_chains)
        for i, chain in enumerate(chains):
            if i < n_good_chains:
                prob = 1 / n_good_chains
            else:
                prob = 0
            chain[5] = prob

    @staticmethod
    def read_has_kmer_around_position(read_kmers, kmer, position, threshold=1):
        if kmer in read_kmers[position-threshold: position+threshold+1]:
            return True
        return False

    @staticmethod
    def align_nodes_to_read(reverse_index, graph, linear_ref, read_ref_pos, read_kmers, read_length=150, previously_aligned_nodes=[]):
        # Find all pairs of variant nodes in this area of graph
        ref_nodes_within_read = linear_ref.get_nodes_between_offset(read_ref_pos, read_ref_pos+read_length)
        variant_nodes = [graph.adj_list[node] for node in ref_nodes_within_read if len(graph.adj_list[node]) >= 2]
        #logging.info("Variant nodes within read: %s" % variant_nodes)

        counts = []
        for variant_node_pair in variant_nodes:
            assert len(variant_node_pair) == 2, "Only biallelic supported now. Nodes: %s" % variant_node_pair
            if variant_node_pair[0] in previously_aligned_nodes or variant_node_pair[1] in previously_aligned_nodes:
                #logging.info("  Ignoring check, since one of nodes is already hit by kmer")
                if variant_node_pair[0] in previously_aligned_nodes:
                    #logging.info("    Added %d" % variant_node_pair[0])
                    counts.append(variant_node_pair[0])
                if variant_node_pair[1] in previously_aligned_nodes:
                    counts.append(variant_node_pair[1])
                    #logging.info("    Added %d" % variant_node_pair[1])
                continue

            for variant_node in variant_node_pair:
                #logging.info("Checking variant node %s" % variant_node)
                graph_kmers, ref_positions = reverse_index.get_node_kmers_and_ref_positions(variant_node)
                for graph_kmer, ref_pos in zip(graph_kmers, ref_positions):
                    #logging.info(" Checking graph kmer %d at position %d" % (graph_kmer, ref_pos))
                    approx_read_position = ref_pos - read_ref_pos
                    if BestChainGenotyper.read_has_kmer_around_position(read_kmers, graph_kmer, approx_read_position):
                        #logging.info("  Match! Graph kmer: %d" % graph_kmer)
                        #logging.info("  Read kmers around pos: %s" % read_kmers[approx_read_position-1: approx_read_position+2])
                        counts.append(variant_node)

            #logging.info(counts)
        return counts


    def _find_best_chains(self):
        j = -1
        for read_kmers, small_read_kmers, smallest_read_kmers in self._read_kmers:
            small_read_kmers = list(small_read_kmers)
            smallest_read_kmers = list(smallest_read_kmers)
            j += 1
            if j % 1000 == 0:
                logging.info("%d reads processed (best chain genotyper)" % j)

            all_nodes = []
            all_ref_offsets = []
            all_read_offsets = []
            for i, hash in enumerate(read_kmers):
                nodes, ref_offsets = self._kmer_index.get_nodes_and_ref_offsets(hash)
                if nodes is None:
                    continue
                all_nodes.append(nodes)
                all_ref_offsets.append(ref_offsets)
                all_read_offsets.append(np.zeros(len(nodes)) + i)

            if len(all_nodes) == 0:
                logging.warning("NO KMER HITS FOR %d" % j)
                continue

            all_nodes = np.concatenate(all_nodes)
            all_ref_offsets = np.concatenate(all_ref_offsets)
            all_read_offsets = np.concatenate(all_read_offsets)

            #best_chain_nodes = BestChainGenotyper.get_nodes_in_best_chain(all_nodes, all_ref_offsets)
            all_chains = BestChainGenotyper.get_all_chains_with_scores(all_nodes, all_ref_offsets, all_read_offsets, self._reference_kmers, set(small_read_kmers), short_k=self._reference_k)
            BestChainGenotyper.assign_probabilities_to_chains(all_chains)
            best_chain_nodes = all_chains[0][0]
            best_chain_ref_pos = all_chains[0][4]
            best_chain_prob = all_chains[0][5]

            counting_nodes = ""  # Simply for logging
            if not self._weight_chains_by_probabilities:
                if self._align_nodes_to_reads:
                    matching_nodes = BestChainGenotyper.align_nodes_to_read(self._align_nodes_to_reads, self._graph, self._linear_path, best_chain_ref_pos, smallest_read_kmers, 150, set(best_chain_nodes))
                    for node in matching_nodes:
                        self._node_counts[node] += best_chain_prob
                    counting_nodes = matching_nodes
                    #best_chain_nodes = matching_nodes

                elif self._remap_best_chain_to is not None:
                    all_nodes_remapped = []
                    all_ref_offsets_remapped = []
                    assert len(small_read_kmers) > 0
                    for hash in small_read_kmers:
                        nodes_remapped, ref_offsets_remapped = self._remap_best_chain_to.get_nodes_and_ref_offsets(hash)
                        if nodes_remapped is None:
                            continue
                        all_nodes_remapped.append(nodes_remapped)
                        all_ref_offsets_remapped.append(ref_offsets_remapped)

                    if len(all_nodes_remapped) == 0:
                        all_nodes_remapped = []
                    else:
                        try:
                            all_ref_offsets_remapped = np.concatenate(all_ref_offsets_remapped)
                            all_nodes_remapped = np.concatenate(all_nodes_remapped)
                        except ValueError:
                            logging.error(all_ref_offsets_remapped)
                            logging.error(all_nodes_remapped)
                            raise

                        # Filter out only those matching this chain
                        try:
                            all_nodes_remapped = all_nodes_remapped[np.where((all_ref_offsets_remapped > best_chain_ref_pos) & (all_ref_offsets_remapped < best_chain_ref_pos + 150))[0]]
                        except TypeError:
                            logging.error(all_ref_offsets_remapped)
                            logging.error(best_chain_ref_pos)
                            raise

                    all_nodes_remapped = np.unique(all_nodes_remapped)
                    for node in all_nodes_remapped:
                        self._node_counts[node] += 1
                    best_chain_nodes = all_nodes_remapped

                else:
                    for node in best_chain_nodes:
                        self._node_counts[node] += 1
            else:
                for chain in all_chains:
                    prob = chain[5]
                    if prob == 0:
                        continue
                    assert prob >= 0
                    assert prob <= 1
                    for node in chain[0]:
                        self._node_counts[node] += prob

            self.write_alignment(j, best_chain_ref_pos, all_chains[0][3] * 150, chromosome=1, nodes=best_chain_nodes, chains=all_chains, counting_nodes=counting_nodes, best_chain_prob=best_chain_prob)
            if self._truth_alignments != None:
                best_chain_match = False
                match = False
                true_position = self._truth_alignments.positions[j]
                for node in all_nodes:
                    if abs(true_position - self._linear_path.get_offset_at_node(node)) <= 1000:
                        match = True
                        break

                for node in best_chain_nodes:
                    if abs(true_position - self._linear_path.get_offset_at_node(node)) <= 1000:
                        best_chain_match = True
                        break

                if best_chain_match:
                    self._best_chain_matches.append(j)
                else:
                    logging.info("============ CHAIN MISSING: %d ==========0" % j)
                    logging.info("Correct offset: %d" % true_position)
                    logging.info(read_kmers)
                    logging.info("Nodes:        %s" % all_nodes)
                    logging.info("Ref offsets:  %s" % all_ref_offsets)
                    logging.info("Read offsets: %s" % all_read_offsets)
                    logging.info("Best chain:   %s" % best_chain_nodes)
                    logging.info("Best chain score:   %.5f" % all_chains[0][3])
                    logging.info("")

                if match:
                    self._has_correct_chain.append(j)
                else:
                    self._has_not_correct_chain.append(j)

                if j == 2908 and False:
                    break

    def get_counts(self):
        self._find_best_chains()

    def genotype(self):
        if self._truth_alignments is not None:
            logging.info("N correct chains found: %d" % len(self._has_correct_chain))
            logging.info("N not correct chains found: %d" % len(self._has_not_correct_chain))
            logging.info("N best chain is correct: %d" % len(self._best_chain_matches))

        count_genotyper = CountGenotyper(self, self._graph, self._sequence_graph, self._vcf_file_name, self._linear_path)
        count_genotyper.genotype()

    def get_node_count(self, node):
        return math.floor(self._node_counts[node])



