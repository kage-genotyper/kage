import logging
from typing import List

import numpy as np
from graph_kmer_index import ReverseKmerIndex
from shared_memory_wrapper import from_file
from obgraph.variant_to_nodes import VariantToNodes


class TrickyVariants:
    properties = {"tricky_variants"}

    def __init__(self, tricky_variants):
        self.tricky_variants = tricky_variants

    @classmethod
    def from_file(cls, file_name):
        return cls(np.load(file_name))

    def to_file(self, file_name):
        np.save(file_name, self.tricky_variants)

    def is_tricky(self, id):
        return self.tricky_variants[id] == 1

    def add(self, other: 'TrickyVariants'):
        self.tricky_variants = np.logical_or(self.tricky_variants, other.tricky_variants)


def find_variants_with_nonunique_kmers(args):
    output = np.zeros(len(args.variant_to_nodes.ref_nodes), dtype=np.uint8)
    n_filtered = 0
    n_sharing_kmers = 0

    for i, (ref_node, var_node) in enumerate(zip(args.variant_to_nodes.ref_nodes, args.variant_to_nodes.var_nodes)):
        reference_kmers = args.reverse_kmer_index.get_node_kmers(ref_node)
        variant_kmers = args.reverse_kmer_index.get_node_kmers(var_node)

        if i % 1000 == 0:
            logging.info("%d variants processed, %d filtered, %d sharing kmers" % (i, n_filtered, n_sharing_kmers))

        frequencies_ref = np.array([args.population_kmer_index.get_frequency(k) - 1 for k in reference_kmers])
        frequencies_var = np.array([args.population_kmer_index.get_frequency(k) - 1 for k in variant_kmers])
        # print(frequencies_ref, frequencies_var)
        # if sum(frequencies_ref) > 0 or sum(frequencies_var) > 0:
        if np.all(frequencies_ref > 0) or np.all(frequencies_var > 0):
            n_filtered += 1
            output[i] = 1
        elif len(set(reference_kmers).intersection(variant_kmers)) > 0:
            n_sharing_kmers += 1
            output[i] = 1

    TrickyVariants(output).to_file(args.out_file_name)
    # np.save(args.out_file_name, output)
    logging.info("Saved array with variants to %s" % args.out_file_name)


def find_tricky_variants(args):
    variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
    # model = GenotypeNodeCountModel.from_file(args.node_count_model)
    # model = NodeCountModelAdvanced.from_file(args.node_count_model)
    model = from_file(args.node_count_model)
    reverse_index = ReverseKmerIndex.from_file(args.reverse_kmer_index)

    tricky_variants = np.zeros(len(variant_to_nodes.ref_nodes + 1), dtype=np.uint32)

    n_tricky_model = 0
    n_tricky_kmers = 0
    n_nonunique = 0

    max_counts_model = args.max_counts_model

    for variant_id in range(0, len(variant_to_nodes.ref_nodes)):
        if variant_id % 100000 == 0:
            logging.info(
                "%d variants processed, %d tricky due to model, %d tricky due to kmers. N non-unique filtered: %d"
                % (variant_id, n_tricky_model, n_tricky_kmers, n_nonunique)
            )

        ref_node = variant_to_nodes.ref_nodes[variant_id]
        var_node = variant_to_nodes.var_nodes[variant_id]

        # model_counts_ref = 1 + model.certain[ref_node] + model.frequencies[ref_node]
        # model_counts_var = 1 + model.certain[var_node] + model.frequencies[var_node]

        if args.only_allow_unique:
            # if model.counts_homo_ref[var_node] > 0 or model.counts_homo_alt[ref_node] > 0:
            if model.has_duplicates(ref_node) or model.has_duplicates(var_node):
                # if model_counts_ref > 1 or model_counts_var > 1:
                n_nonunique += 1
                tricky_variants[variant_id] = 1

        # if model_counts_ref[2] > max_counts_model and model_counts_var[2] > max_counts_model:
        # if model_counts_ref[2] < model_counts_ref[1] * 1.1 or model_counts_var[2] < model_counts_var[1] * 1.1:
        m = args.max_counts_model
        if model.has_no_data(ref_node) or model.has_no_data(var_node):
            # logging.warning(model_counts_ref)
            # logging.warning(model_counts_ref)
            tricky_variants[variant_id] = 1
            # print(model[1][ref_node], model[1][var_node])
            n_tricky_model += 1
        else:
            reference_kmers = set(reverse_index.get_node_kmers(ref_node))
            variant_kmers = set(reverse_index.get_node_kmers(var_node))
            if len(reference_kmers.intersection(variant_kmers)) > 0:
                # logging.warning("-----\nKmer crash on variant %d \n Ref kmers: %s\n Var kmers: %s" % (variant_id, reference_kmers, variant_kmers))
                tricky_variants[variant_id] = 1
                n_tricky_kmers += 1

    logging.info(
        "Stats: %d tricky due to model, %d tricky due to kmers. N non-unique filtered: %d"
        % (n_tricky_model, n_tricky_kmers, n_nonunique)
    )

    TrickyVariants(tricky_variants).to_file(args.out_file_name)
    # np.save(args.out_file_name, tricky_variants)
    logging.info("Wrote tricky variants to file %s" % args.out_file_name)



