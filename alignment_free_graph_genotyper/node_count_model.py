import numpy as np
from Bio.Seq import Seq
import logging
from obgraph import Graph, VariantNotFoundException
from obgraph.genotype_matrix import GenotypeMatrix
from graph_kmer_index import ReverseKmerIndex, KmerIndex
import time
from alignment_free_graph_genotyper import cython_chain_genotyper
import itertools

class GenotypeNodeCountModel:
    properties = {"counts_homo_ref", "counts_homo_alt", "counts_hetero"}
    def __init__(self, counts_homo_ref=None, counts_homo_alt=None, counts_hetero=None):
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


class NodeCountModelAlleleFrequencies:
    properties = {"allele_frequencies", "allele_frequencies_squared"}
    def __init__(self, allele_frequencies=None, allele_frequencies_squared=None, average_coverage=1):
        self.allele_frequencies = allele_frequencies
        self.allele_frequencies_squared = allele_frequencies_squared

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(data["allele_frequencies"], data["allele_frequencies_squared"])

    def to_file(self, file_name):
        np.savez(file_name, allele_frequencies=self.allele_frequencies,
                 allele_frequencies_squared=self.allele_frequencies_squared)


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


class GenotypeModelCreatorFromTransitionProbabilities:
    def __init__(self, graph, genotype_matrix: GenotypeMatrix, variant_to_nodes, node_to_variants, mapping_index: KmerIndex, kmer_index: KmerIndex,  max_node_id):
        self.graph = graph
        self.kmer_index = kmer_index
        self.variant_to_nodes = variant_to_nodes
        self.node_to_variants = node_to_variants
        self.mapping_index = mapping_index
        self.genotype_matrix = genotype_matrix

        self.counts_homo_ref = np.zeros(max_node_id)
        self.counts_homo_alt = np.zeros(max_node_id)
        self.counts_hetero = np.zeros(max_node_id)

    def _get_variant_info_for_kmer(self, kmer):
        node_groups = self.kmer_index.get_grouped_nodes(kmer, 10000000)

        assert len(node_groups) > 0, "No index hits for kmer %d" % kmer

        variants = []
        for nodes in node_groups:
            possible_variants = []
            for node in nodes:

                variant_id = self.node_to_variants.get_variant_at_node(node)
                if variant_id is None:
                    continue

                possible_genotypes = [3]  # possible genotypes at this variant where this node is included
                if self.variant_to_nodes.ref_nodes[variant_id] == node:
                    possible_genotypes.append(1)
                else:
                    possible_genotypes.append(2)

                allele_frequency = self.graph.get_node_allele_frequency(node)

                possible_variants.append((variant_id, allele_frequency, possible_genotypes))

            if len(possible_variants) == 0:
                # covers no variants, add a ref-entry
                variants.append([(None, 1.0, None)])
            else:
                # choose the variant with lowest allele frequency
                #variant = sorted(possible_variants, key=lambda v: v[1])[0]
                variants.append(possible_variants)

        if kmer == 965251971216330571:
            print("Variants: %s" % variants)

        return variants

    def _process_variant(self, kmer_entries_in_population, variant_node, variant_id_for_variant_node, variant_node_is_ref):
        # expected node counts for the variant_node
        counts_homo_ref = 0
        counts_homo_alt = 0
        counts_hetero = 0

        #for variant_id, _, is_ref_node in kmer_entries_in_population:
        # variants is a list of variants for a given kmer entry/location in the population
        for variants in kmer_entries_in_population:

            #print(variant_id, is_ref_node, variant_id_for_variant_node, variant_node_is_ref)
            if len(variants) == 1 and variants[0][0] is None:
                # this is an entry on the ref path, meaning it will contribute with a factor of 2.0 (both haplotypes in any individual will have this kmer)
                #print("Contributing from ref to node %d" % variant_id_for_variant_node)
                counts_hetero += 2
                counts_homo_ref += 2
                counts_homo_alt += 2
            elif variant_id_for_variant_node in [variant_info[0] for variant_info in variants]:
                continue  # skip, add these in the end instead
                if variant_node == 25:
                    print("Contributing from self to node %d" % variant_id_for_variant_node)
                # our kmer is sampled from the same node as the variant,
                # if we are hetero at this variant, we will get 1 in count since only one haplotype will have the kmer
                # if we are homo ref at variant, both haplotypes at variant_id will give, and so on
                counts_hetero += 1
                common_variant = [v for v in variants if v[0] == variant_id_for_variant_node][0]
                variant_node_is_ref = common_variant[2]
                if variant_node_is_ref:
                    counts_homo_ref += 2
                else:
                    counts_homo_alt += 2
            else:
                # this is an entry on a variant node (either ref or alt).
                # this entry may cover multiple variants, to simplify things only choose one of them when computing the transition probabilities/similarity
                # estimate the proportion of individuals having this node also have the variant_node
                # a is the variant at variant_id (in the population)
                # b is the variant we are estimating counts for
                # 1= homo ref, 2=homo alt, 3=hetero

                if len(variants) == 1:
                    chosen_variant = variants[0]
                else:
                    # pick the one with lowest allele freq
                    chosen_variant = sorted(variants, key=lambda v: v[1])[0]

                variant_id, _, possible_genotypes_a = chosen_variant
                possible_genotypes_b = [1, 2, 3]

                # permute all genotypes at all variants at location a
                # we want to find the prob that the individual with given genotype at variant b has the a-variants with a-genotypes

                def permute_variants(variants):
                    flat_variants = []
                    for variant in variants:
                        v = []
                        for genotype in variant[2]:
                            v.append((variant[0], genotype))
                        flat_variants.append(v)

                    permuted_variants = list(itertools.product(*flat_variants))
                    #print("Input: %s\n  Flat: %s\n   Permuted: %s" % (variants, flat_variants, permuted_variants))
                    return permuted_variants

                permuted_variants = permute_variants(variants)
                if len(permuted_variants) > 100:
                    permuted_variants = permuted_variants[0:100]

                for genotype_b in possible_genotypes_b:
                    for variant_series in permuted_variants:
                        if 3 in [v[1] for v in variant_series]:  # one of the variants is heterozygous, the individual can only have this kmer once
                            factor = 1
                        else:
                            factor = 2

                        transition_prob = self.genotype_matrix.get_transition_prob_from_single_to_multiple_variants(variant_id_for_variant_node, genotype_b, variant_series)

                        #print("Transition prob from %d/%d to %s: %.4f" % (variant_id_for_variant_node, genotype_b, variant_series, transition_prob))
                        if genotype_b == 1:
                            counts_homo_ref += factor * transition_prob
                        elif genotype_b == 2:
                            counts_homo_alt += factor * transition_prob
                        else:
                            counts_hetero += factor * transition_prob

                    """
                for genotype_a in possible_genotypes_a:
                    if genotype_a == 1 or genotype_a == 2:
                        factor = 2
                    else:
                        factor = 1

                    for genotype_b in possible_genotypes_b:
                        transition_prob = self.genotype_matrix.get_transition_prob(variant_id_for_variant_node, variant_id, genotype_b, genotype_a)

                        if variant_node == 25:
                            print("Kmer entries: %s" % kmer_entries_in_population)
                            print("  Transition prob from genotype %d at %d to %d at %d: %.5f" % (genotype_b, variant_id_for_variant_node, genotype_a, variant_id, transition_prob))

                        if genotype_b == 1:
                            counts_homo_ref += factor * transition_prob
                        elif genotype_b == 2:
                            counts_homo_alt += factor * transition_prob
                        else:
                            counts_hetero += factor * transition_prob
                """

        self.counts_hetero[variant_node] += counts_hetero
        self.counts_homo_ref[variant_node] += counts_homo_ref
        self.counts_homo_alt[variant_node] += counts_homo_alt


    def get_node_counts(self):
        unique_kmers = np.unique(self.kmer_index._kmers)

        for i, kmer in enumerate(unique_kmers):
            kmer = int(kmer)
            if i % 10000 == 0:
                logging.info("%d/%d kmers processed" % (i, len(unique_kmers)))
                logging.info("Sum of counts: %d, %d, %d" % (np.sum(self.counts_homo_ref), np.sum(self.counts_homo_alt), np.sum(self.counts_hetero)))

            kmer_entries_in_population = self._get_variant_info_for_kmer(kmer)  # locations in graph that this kmer exist
            #print("Population kmers: %s" % kmer_entries_in_population)

            variant_hits = self.mapping_index.get_grouped_nodes(kmer, max_hits=1000000000)
            if variant_hits is None:
                continue

            #print("Variants hits: %s" % variant_hits)
            for hit in variant_hits:
                #print(type(hit))
                # for every variant node, estimate the count this kmer will give
                for node in hit:
                    #print("   Node: %s" % node)
                    variant = self.node_to_variants.get_variant_at_node(node)
                    if node == 6:
                        print(node, variant)
                    if variant is not None:
                        is_ref = False
                        if self.variant_to_nodes.ref_nodes[variant] == node:
                            is_ref = True
                        self._process_variant(kmer_entries_in_population, node, variant, is_ref)
                    else:
                        print("Node %d has no variant" % node)


        self.counts_hetero[self.variant_to_nodes.var_nodes] += 1
        self.counts_hetero[self.variant_to_nodes.ref_nodes] += 1
        self.counts_homo_alt[self.variant_to_nodes.var_nodes] += 2
        self.counts_homo_ref[self.variant_to_nodes.ref_nodes] += 2




class _GenotypeModelCreatorFromTransitionProbabilities:
    def __init__(self, genotype_matrix, variant_to_nodes, node_to_variants, reverse_index: ReverseKmerIndex, kmer_index: KmerIndex, variant_start_id, variant_end_id, max_node_id, genotype_frequencies):
        self.kmer_index = kmer_index
        self.variant_to_nodes = variant_to_nodes
        self.node_to_variants = node_to_variants
        self.reverse_index = reverse_index
        self.variant_start_id = variant_start_id
        self.variant_end_id = variant_end_id

        self.counts_homo_ref = np.zeros(max_node_id)
        self.counts_homo_alt = np.zeros(max_node_id)
        self.counts_hetero = np.zeros(max_node_id)

    def _get_other_variant_hits(self, kmer_node_hits, variant_id, ref_node, var_node, genotype):
        hits = []  # (variant id, is variant node true/false)
        for nodes in kmer_node_hits:

            if ref_node in nodes or var_node in nodes:
                # this hit is the actual variant we are analysing, ignore
                continue

            if len(nodes) == 0:
                continue

            variants_detected = []
            for node in nodes:
                variant_id_at_hit = self.node_to_variants.get_variant_at_node(node)
                if variant_id is None:
                    continue


                # is this a hit on variant node or ref node?
                is_variant_node = True
                if self.node_to_variants.ref_node[node] == variant_id_at_hit:
                    is_variant_node = False

                variants_detected.append((variant_id, is_variant_node))

            # among the variants covered by this hit, estimate the probability that the individual

    def _get_expected_count_from_graph(self, variant_id, ref_node, var_node, variant_genotype):
        # given this variant id and genotype, find expected counts from rest of graph
        counts_ref = 0
        counts_var = 0

        ref_kmers = self.reverse_index.get_node_kmers(ref_node)
        var_kmers = self.reverse_index.get_node_kmers(var_node)

        for kmer in ref_kmers, var_kmers:
            other_hits = self.kmer_index.get_grouped_nodes(kmer)
            other_hits



    def process_variant(self, variant_id):
        ref_node = self.variant_to_nodes.ref_nodes[variant_id]
        var_node = self.variant_to_nodes.var_nodes[variant_id]

        for genotype in [1, 2, 3]:
            # find expected counts for ref_node and var_node given genotype
            if genotype == 1:
                ref_node_count = 2
                var_node_count = 0
            elif genotype == 2:
                ref_node_count = 0
                var_node_count = 2
            else:
                ref_node_count = 1
                var_node_count = 1

            expected_from_graph_ref, expected_from_graph_var = self._get_expected_count_from_graph(variant_id, ref_node, var_node, genotype)

    def get_node_counts(self):
        for i, variant_id in enumerate(range(self.variant_start_id, self.variant_end_id)):
            if i % 25000 == 0:
                logging.info("%d/%d variants processed" % (i, self.variant_end_id - self.variant_start_id))

            # reference_node, variant_node = self.graph.get_variant_nodes(variant)
            reference_node = self.variant_to_nodes.ref_nodes[variant_id]
            variant_node = self.variant_to_nodes.var_nodes[variant_id]

            if reference_node == 0 or variant_node == 0:
                continue

            self.process_variant(reference_node, variant_node)

        return self.node_counts_following_node, self.node_counts_not_following_node


class NodeCountModelCreatorFromNoChaining:
    def __init__(self, kmer_index: KmerIndex, reverse_index: ReverseKmerIndex, variant_to_nodes, variant_start_id, variant_end_id, max_node_id,
                 scale_by_frequency=False, allele_frequency_index=None, haplotype_matrix=None, node_to_variants=None):
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

        self.haplotype_matrix = haplotype_matrix
        self.node_to_variants = node_to_variants
        self.variant_to_nodes = variant_to_nodes

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
                    if self.haplotype_matrix is not None:
                        ref_offset = ref_offsets[index]
                        hit_nodes = nodes[np.where(ref_offsets == ref_offset)]
                        allele_frequency = self.haplotype_matrix.get_allele_frequency_for_nodes(hit_nodes, self.node_to_variants, self.variant_to_nodes)
                    elif self._allele_frequency_index is None:
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

            expected_count_following += 1.0


            self.node_counts_following_node[node] += expected_count_following
            self.node_counts_not_following_node[node] += expected_count_not_following

    def create_model(self):
        for i, variant_id in enumerate(range(self.variant_start_id, self.variant_end_id)):
            if i % 25000 == 0:
                logging.info("%d/%d variants processed" % (i, self.variant_end_id-self.variant_start_id))

            #reference_node, variant_node = self.graph.get_variant_nodes(variant)
            reference_node = self.variant_to_nodes.ref_nodes[variant_id]
            variant_node = self.variant_to_nodes.var_nodes[variant_id]

            if reference_node == 0 or variant_node == 0:
                continue

            self.process_variant(reference_node, variant_node)

    def get_results(self):
        return self.node_counts_following_node, self.node_counts_not_following_node


class NodeCountModelCreatorFromNoChainingOnlyAlleleFrequencies(NodeCountModelCreatorFromNoChaining):
    def __init__(self, kmer_index: KmerIndex, reverse_index: ReverseKmerIndex, variant_to_nodes, variant_start_id,
                 variant_end_id, max_node_id,
                 scale_by_frequency=False, allele_frequency_index=None, haplotype_matrix=None, node_to_variants=None):
        self.kmer_index = kmer_index
        self.variant_to_nodes = variant_to_nodes
        self.reverse_index = reverse_index
        self.variant_start_id = variant_start_id
        self.variant_end_id = variant_end_id

        self._allele_frequencies_summed = np.zeros(max_node_id + 1, dtype=np.float)
        self._allele_frequencies_sum_of_squares = np.zeros(max_node_id + 1, dtype=np.float)

        self._allele_frequency_index = allele_frequency_index
        if self._allele_frequency_index is not None:
            logging.info("Will fetch allele frequencies from allele frequency index")

        self.haplotype_matrix = haplotype_matrix
        self.node_to_variants = node_to_variants
        self.variant_to_nodes = variant_to_nodes

    def process_variant(self, reference_node, variant_node):
        for node in (reference_node, variant_node):
            allele_frequencies_found = []

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
                    # logging.warning("Found not index hits for kmer %d" % kmer)
                    continue

                n_hits = 0
                for index in unique_indexes:
                    # do not add count for the actual kmer we are searching for, we add 1 for this in the end
                    if self.haplotype_matrix is not None:
                        ref_offset = ref_offsets[index]
                        hit_nodes = nodes[np.where(ref_offsets == ref_offset)]
                        allele_frequency = self.haplotype_matrix.get_allele_frequency_for_nodes(hit_nodes,
                                                                                                self.node_to_variants,
                                                                                                self.variant_to_nodes)
                    elif self._allele_frequency_index is None:
                        allele_frequency = allele_frequencies[index]  # fetch from graph
                    else:
                        # get the nodes belonging to this ref offset
                        ref_offset = ref_offsets[index]
                        hit_nodes = nodes[np.where(ref_offsets == ref_offset)]
                        # allele frequency is the lowest allele frequency for these nodes
                        allele_frequency = np.min(self._allele_frequency_index[hit_nodes])

                    if ref_offsets[index] != ref_pos:
                        allele_frequencies_found.append(allele_frequency)


            allele_frequencies_found = np.array(allele_frequencies_found)
            self._allele_frequencies_summed[node] = np.sum(allele_frequencies_found)
            self._allele_frequencies_sum_of_squares[node] = np.sum(allele_frequencies_found**2)

    def get_results(self):
        return self._allele_frequencies_summed, self._allele_frequencies_sum_of_squares


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