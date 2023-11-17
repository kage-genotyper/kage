from collections import defaultdict

import bionumpy as bnp
import numpy as np
import pickle

from kage.analysis.genotype_accuracy import IndexedGenotypes2
from kage.io import VcfWithSingleIndividualBuffer

from ..indexing.index_bundle import IndexBundle
from graph_kmer_index import kmer_hash_to_sequence

def pretty_variant(variant):
    return f"{variant.chromosome}:{variant.position} {variant.ref_seq.to_string()}/{variant.alt_seq.to_string()} {variant.genotype}"

class Debugger:
    def __init__(self, genotype_report, kage_index, truth_vcf, genotypes_vcf, node_counts, probs, count_probs, numeric_genotypes=None):
        self.report = genotype_report
        self.kage_index = kage_index
        self.truth_vcf = truth_vcf
        self.genotypes_vcf = genotypes_vcf
        self.node_counts = node_counts
        self.helper = self.kage_index["helper_variants"]
        self.combination_matrix = self.kage_index["combination_matrix"]
        self.count_model = self.kage_index["count_model"]
        self.variant_to_nodes = self.kage_index["variant_to_nodes"]
        self.tricky_variants = self.kage_index["tricky_variants"]
        self.tricky_alleles = self.kage_index["tricky_alleles"]
        self.tricky_ref, self.tricky_alt = self.tricky_alleles
        self.kmer_index = self.kage_index["kmer_index"]
        self.probs = probs
        self.count_probs = count_probs
        self.numeric_genotypes = numeric_genotypes
        self.orig_count_model = None
        if "orig_count_model" in self.kage_index:
            self.orig_count_model = self.kage_index["orig_count_model"]

        #self.genotypes = bnp.open(genotypes_vcf, buffer_type=bnp.io.vcf_buffers.PhasedVCFMatrixBuffer).read()
        #self.genotypes = bnp.open(genotypes_vcf, buffer_type=VcfWithSingleIndividualBuffer).read()
        #self.truth = bnp.open(truth_vcf, buffer_type=VcfWithSingleIndividualBuffer).read()
        self.genotypes = IndexedGenotypes2.from_biallelic_vcf(genotypes_vcf)
        self.truth = IndexedGenotypes2.from_biallelic_vcf(truth_vcf)
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

    def print_variant_info(self, variant_id, variant_number, with_helper=True):
        id = variant_number
        ref_node = self.variant_to_nodes.ref_nodes[id]
        var_node = self.variant_to_nodes.var_nodes[id]
        print("-----\n", variant_id, variant_number)
        print("Nodes: ", ref_node, var_node)
        print("Trtuh ", id, self.truth[variant_id].genotype)
        print("Called", self.genotypes[variant_id].genotype)
        print(f"Node counts: {self.node_counts[ref_node]}/{self.node_counts[var_node]}")
        if self.tricky_variants.is_tricky(id):
            print("IS TRICKY VARIANT")
        if self.tricky_ref.is_tricky(id):
            print("REF IS TRICKY")
        if self.tricky_alt.is_tricky(id):
            print("ALT IS TRICKY")

        print("Count model ref", self.count_model[0].describe_node(id))
        print("Count model alt", self.count_model[1].describe_node(id))
        if self.orig_count_model is not None:
            print("Orig count model ref", self.orig_count_model.describe_node(ref_node))
            print("Orig count model alt", self.orig_count_model.describe_node(var_node))

        print("Kmer ref: ", ",".join([str(k) + "," + kmer_hash_to_sequence(k, 31) for k in self.reverse_kmer_index[ref_node]]))
        print("Kmer alt: ", ",".join([str(k) + "," + kmer_hash_to_sequence(k, 31) for k in self.reverse_kmer_index[var_node]]))
        print("Probs", self.probs[id])
        print("Count probs", self.count_probs[id])
        print("Numeric genotype", self.numeric_genotypes[id])
        if with_helper:
            helper = self.helper[id]
            print("Most similar variant: %d" % helper)
            print("Combination matrix: \n%s" % self.combination_matrix[id])
            print("Genotype probs most similar (log): %s" % self.probs[helper])
            #print("--Helper variant--")
            #self.print_variant_info(helper, with_helper=False)

    def run(self):
        for type in ["false_positives", "false_negatives"]:
            print(type.upper() + "----------_")
            for variant_id, variant_number in self.report[type]:
                if len(variant_id) >= 50:
                    self.print_variant_info(variant_id, variant_number)
                    print()
                    print()



def debug_cli(args):
    debugger = Debugger(pickle.load(open(args.report, "rb")),
                        IndexBundle.from_file(args.index), args.truth, args.genotypes,
                        np.load(args.node_counts),
                        np.load(args.probs),
                        np.load(args.count_probs),
                        np.load(args.numeric_genotypes))
    debugger.run()