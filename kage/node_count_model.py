import numpy as np
import logging
from graph_kmer_index import ReverseKmerIndex, KmerIndex


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

        counts_hetero[var_nodes] = (
            model.node_counts_following_node[var_nodes]
            + model.node_counts_not_following_node[var_nodes]
        )
        counts_hetero[ref_nodes] = (
            model.node_counts_following_node[ref_nodes]
            + model.node_counts_not_following_node[ref_nodes]
        )

        return cls(counts_homo_ref, counts_homo_alt, counts_hetero)

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(
            data["counts_homo_ref"], data["counts_homo_alt"], data["counts_hetero"]
        )

    def to_file(self, file_name):
        np.savez(
            file_name,
            counts_homo_ref=self.counts_homo_ref,
            counts_homo_alt=self.counts_homo_alt,
            counts_hetero=self.counts_hetero,
        )


class NodeCountModelAlleleFrequencies:
    properties = {"allele_frequencies", "allele_frequencies_squared"}

    def __init__(
        self,
        allele_frequencies=None,
        allele_frequencies_squared=None,
        average_coverage=1,
    ):
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
        np.savez(
            file_name,
            allele_frequencies=self.allele_frequencies,
            allele_frequencies_squared=self.allele_frequencies_squared,
        )


class NodeCountModelAdvanced:
    properties = {
        "frequencies",
        "frequencies_squared",
        "certain",
        "frequency_matrix",
        "has_too_many",
    }

    def __init__(
        self,
        frequencies=None,
        frequencies_squared=None,
        certain=None,
        frequency_matrix=None,
        has_too_many=None,
    ):
        self.frequencies = frequencies
        self.frequencies_squared = frequencies_squared
        self.certain = certain
        self.frequency_matrix = frequency_matrix
        self.has_too_many = has_too_many

    @classmethod
    def from_dict_of_frequencies(cls, frequencies, n_nodes):
        model = cls.create_empty(n_nodes)
        for node, frequencies in frequencies.items():
            model.frequencies[node] = np.sum(frequencies)
            model.frequencies_squared[node] = np.sum(np.array(frequencies) ** 2)
            model.certain[node] = len(np.where(frequencies == 1)[0])
            if len(frequencies) > 5:
                model.has_too_many[node] = True
            else:
                model.frequency_matrix[node, 0 : len(frequencies)] = frequencies

        return model

    def describe_node(self, node):
        description = "Frequencies: %.3f, certain: %d, has too many? %s" % (
            self.frequencies[node],
            self.certain[node],
            self.has_too_many[node],
        )
        return description

    @classmethod
    def create_empty(cls, max_node_id):
        frequencies = np.zeros(max_node_id + 1, dtype=float)
        frequencies_squared = np.zeros(max_node_id + 1, dtype=float)
        certain = np.zeros(max_node_id + 1, dtype=float)
        frequency_matrix = np.zeros((max_node_id + 1, 5), dtype=float)
        has_too_many = np.zeros(max_node_id + 1, dtype=bool)
        return cls(
            frequencies, frequencies_squared, certain, frequency_matrix, has_too_many
        )

    def __add__(self, other):
        self.frequencies += other.frequencies
        self.frequencies_squared += other.frequencies_squared
        self.certain += other.certain
        self.frequency_matrix += other.frequency_matrix
        self.has_too_many += other.has_too_many
        return self

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(
            data["frequencies"],
            data["frequencies_squared"],
            data["certain"],
            data["frequency_matrix"].astype(np.float32),
            data["has_too_many"],
        )

    def to_file(self, file_name):
        np.savez(
            file_name,
            frequencies=self.frequencies,
            frequencies_squared=self.frequencies_squared,
            certain=self.certain,
            frequency_matrix=self.frequency_matrix,
            has_too_many=self.has_too_many,
        )


class NodeCountModel:
    def __init__(
        self,
        node_counts_following_node,
        node_counts_not_following_node,
        average_coverage=1,
    ):
        self.node_counts_following_node = node_counts_following_node
        self.node_counts_not_following_node = node_counts_not_following_node

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(
            data["node_counts_following_node"], data["node_counts_not_following_node"]
        )

    def to_file(self, file_name):
        np.savez(
            file_name,
            node_counts_following_node=self.node_counts_following_node,
            node_counts_not_following_node=self.node_counts_not_following_node,
        )


class NodeCountModelCreatorAdvanced:
    def __init__(
        self,
        kmer_index: KmerIndex,
        reverse_index: ReverseKmerIndex,
        variant_to_nodes,
        variant_start_id,
        variant_end_id,
        max_node_id,
        scale_by_frequency=False,
        allele_frequency_index=None,
        haplotype_matrix=None,
        node_to_variants=None,
    ):
        self.kmer_index = kmer_index
        self.variant_to_nodes = variant_to_nodes
        self.reverse_index = reverse_index
        self.variant_start_id = variant_start_id
        self.variant_end_id = variant_end_id

        self._frequencies = np.zeros(max_node_id + 1, dtype=float)
        self._frequencies_squared = np.zeros(max_node_id + 1, dtype=float)
        self._certain = np.zeros(max_node_id + 1, dtype=float)
        self._frequency_matrix = np.zeros((max_node_id + 1, 5), dtype=float)
        self._has_too_many = np.zeros(max_node_id + 1, dtype=bool)

        self._allele_frequency_index = allele_frequency_index
        if self._allele_frequency_index is not None:
            logging.info("Will fetch allele frequencies from allele frequency index")

        self.haplotype_matrix = haplotype_matrix
        self.node_to_variants = node_to_variants
        self.variant_to_nodes = variant_to_nodes

    def process_variant(self, reference_node, variant_node):

        for node in (reference_node, variant_node):
            allele_frequencies_found = []
            # allele_frequencies_found = [0.01]  # a dummy frequency  found

            expected_count_following = 0
            expected_count_not_following = 0
            lookup = self.reverse_index.get_node_kmers_and_ref_positions(node)
            for result in zip(lookup[0], lookup[1]):
                kmer = result[0]
                ref_pos = result[1]
                kmer = int(kmer)
                (
                    nodes,
                    ref_offsets,
                    frequencies,
                    allele_frequencies,
                ) = self.kmer_index.get(kmer, max_hits=1000000)
                if nodes is None:
                    logging.error("!!!!!!!!!!!")
                    logging.error("Kmer %d was not found in whole genome kmers" % kmer)
                    logging.error(
                        "Ref pos for kmer is %d. Variant/refnode: %d/%d"
                        % (ref_pos, variant_node, reference_node)
                    )
                    # raise Exception("Failed. All k mers should exiset in whole genome index")
                    continue

                unique_ref_offsets, unique_indexes = np.unique(
                    ref_offsets, return_index=True
                )

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
                        allele_frequency = (
                            self.haplotype_matrix.get_allele_frequency_for_nodes(
                                hit_nodes, self.node_to_variants, self.variant_to_nodes
                            )
                        )
                    elif self._allele_frequency_index is None:
                        allele_frequency = allele_frequencies[index]  # fetch from graph
                    else:
                        # get the nodes belonging to this ref offset
                        ref_offset = ref_offsets[index]
                        hit_nodes = nodes[np.where(ref_offsets == ref_offset)]
                        # allele frequency is the lowest allele frequency for these nodes
                        allele_frequency = np.min(
                            self._allele_frequency_index[hit_nodes]
                        )

                    if ref_offsets[index] != ref_pos:
                        if allele_frequency == 1:
                            self._certain[node] += 1
                        else:
                            allele_frequencies_found.append(allele_frequency)

            allele_frequencies_found = np.array(allele_frequencies_found)
            self._frequencies[node] = np.sum(allele_frequencies_found)
            self._frequencies_squared[node] = np.sum(allele_frequencies_found ** 2)
            if len(allele_frequencies_found) <= 5:
                self._frequency_matrix[
                    node, 0 : len(allele_frequencies_found)
                ] = allele_frequencies_found
            else:
                self._has_too_many[node] = True

    def create_model(self):
        for i, variant_id in enumerate(
            range(self.variant_start_id, self.variant_end_id)
        ):
            if i % 25000 == 0:
                logging.info(
                    "%d/%d variants processed"
                    % (i, self.variant_end_id - self.variant_start_id)
                )

            # reference_node, variant_node = self.graph.get_variant_nodes(variant)
            reference_node = self.variant_to_nodes.ref_nodes[variant_id]
            variant_node = self.variant_to_nodes.var_nodes[variant_id]

            if reference_node == 0 or variant_node == 0:
                continue

            self.process_variant(reference_node, variant_node)

    def get_results(self):
        return NodeCountModelAdvanced(
            self._frequencies,
            self._frequencies_squared,
            self._certain,
            self._frequency_matrix,
            self._has_too_many,
        )
