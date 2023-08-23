from collections import defaultdict

import bionumpy as bnp
import numpy as np
import pickle
from ..indexing.index_bundle import IndexBundle
from graph_kmer_index import kmer_hash_to_sequence

def pretty_variant(variant):
    return f"{variant.chromosome}:{variant.position} {variant.ref_seq.to_string()}/{variant.alt_seq.to_string()} {variant.genotypes[0]}"

class Debugger:
    def __init__(self, genotype_report, kage_index, truth_vcf, genotypes_vcf, node_counts):
        self.report = genotype_report
        self.kage_index = kage_index
        self.truth_vcf = truth_vcf
        self.genotypes_vcf = genotypes_vcf
        self.node_counts = node_counts
        self.helper = self.kage_index["helper_variants"]
        self.count_model = self.kage_index["count_model"]
        self.variant_to_nodes = self.kage_index["variant_to_nodes"]
        self.tricky_variants = self.kage_index["tricky_variants"]
        self.kmer_index = self.kage_index["kmer_index"]

        self.genotypes = bnp.open(genotypes_vcf, buffer_type=bnp.io.delimited_buffers.PhasedVCFMatrixBuffer).read()
        self.truth = bnp.open(truth_vcf, buffer_type=bnp.io.delimited_buffers.PhasedVCFMatrixBuffer).read()
        self.reverse_kmer_index = self.get_reverse_kmer_index()

    def get_reverse_kmer_index(self):
        index = defaultdict(list)

        #for kmer in self.kmer_index._kmers:
        #    for node in self.kmer_index.get_nodes(kmer):
        #        index[node].append(kmer)
        nodes = self.kmer_index._nodes
        kmers = self.kmer_index._kmers
        for node, kmer in zip(nodes, kmers):
            index[node].append(kmer)

        return index

    def print_variant_info(self, id, with_helper=True):
        ref_node = self.variant_to_nodes.ref_nodes[id]
        var_node = self.variant_to_nodes.var_nodes[id]
        print(id, "Nodes: ", ref_node, var_node)
        print("Called", pretty_variant(self.genotypes[id]))
        print("Trtuh ", pretty_variant(self.truth[id]))
        print(f"Node counts: {self.node_counts[ref_node]}/{self.node_counts[var_node]}")
        if self.tricky_variants.is_tricky(id):
            print("IS TRICKY VARIANT")
        print("Count model ref", self.count_model[0].describe_node(id))
        print("Count model alt", self.count_model[1].describe_node(id))
        print("Kmer ref: ", ",".join([str(k) + "," + kmer_hash_to_sequence(k, 31) for k in self.reverse_kmer_index[ref_node]]))
        print("Kmer alt: ", ",".join([str(k) + "," + kmer_hash_to_sequence(k, 31) for k in self.reverse_kmer_index[var_node]]))
        if with_helper:
            helper = self.helper[id]
            print("--Helper variant--")
            self.print_variant_info(helper, with_helper=False)

    def run(self):
        for type in ["false_positives", "false_negatives"]:
            print(type.upper() + "----------_")
            for id in self.report[type]:
                self.print_variant_info(id)
                print()
                print()



def debug_cli(args):
    debugger = Debugger(pickle.load(open(args.report, "rb")),
                        IndexBundle.from_file(args.index), args.truth, args.genotypes,
                        np.load(args.node_counts))
    debugger.run()