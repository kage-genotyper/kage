import bionumpy as bnp
import numpy as np
import pickle
from ..indexing.index_bundle import IndexBundle

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

        self.genotypes = bnp.open(genotypes_vcf, buffer_type=bnp.io.delimited_buffers.PhasedVCFMatrixBuffer).read()
        self.truth = bnp.open(truth_vcf, buffer_type=bnp.io.delimited_buffers.PhasedVCFMatrixBuffer).read()

    def print_variant_info(self, id, with_helper=True):
        print(id)
        print("Called", pretty_variant(self.genotypes[id]))
        print("Trtuh ", pretty_variant(self.truth[id]))
        ref_node = id * 2
        var_node = id * 2 + 1
        print(f"Node counts: {self.node_counts[ref_node]}/{self.node_counts[var_node]}")
        #print("Count model ref", self.count_model[0].describe_node[id])
        #print("Count model alt", self.count_model[1].describe_node[id])
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