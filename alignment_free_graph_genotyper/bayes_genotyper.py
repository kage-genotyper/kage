from .genotyper import Genotyper
#from scipy.stats import betabinom

class NewBayesGenotyper(Genotyper):
    alpha = 1
    beta = 1
    priors = {"0/0": (0.9, 0.1),
              "0/1": (0.5, 0.5),
              "1/1": (0.1, 0.9)}

    def prob_observing_counts(self, ref_node, variant_node, genotype, type="binomial"):
        ref_count = int(self.get_node_count(ref_node))
        variant_count = int(self.get_node_count(variant_node))

        if genotype == "1/1":
            counts = self._node_count_model.counts_homo_alt
        elif genotype == "0/0":
            counts = self._node_count_model.counts_homo_ref
        elif genotype == "0/1":
            counts = self._node_count_model.counts_hetero
        else:
            raise Exception("Unsupported genotype %s" % genotype)
        alpha, beta = self.priors[genotype]
        if False:
            print(genotype, ref_count, ref_count+variant_count,
                  alpha + counts[ref_node], beta+counts[variant_node],
                  betabinom.pmf(ref_count, ref_count+variant_count,
                                alpha + counts[ref_node], beta+counts[variant_node]))

        return betabinom.pmf(ref_count, ref_count+variant_count,
                             alpha + counts[ref_node], beta+counts[variant_node])

    def _genotype_biallelic_variant(self, reference_node, variant_node, a_priori_homozygous_ref, a_priori_homozygous_alt,
                                    a_priori_heterozygous, debug=False):

        genotypes = ("0/0", "1/1", "0/1")
        priors = (a_priori_homozygous_ref, a_priori_homozygous_alt, a_priori_heterozygous)
        p_counts = [self.prob_observing_counts(reference_node, variant_node, g) for g in genotypes]
        posteriors = [prior*p_count for prior, p_count in zip(priors, p_counts)]
        #print(self.get_node_count(reference_node), self.get_node_count(variant_node))
        #print(["{:.2f}".format(p/sum(posteriors)) for p in posteriors])
        return max(zip(posteriors, genotypes))[1]

    # def genotype(self):
    #     variant_id = -1
    #     # print(self._variants)
    #     variants = list(range(len(self._variants)))
    #     ref_nodes = self._variant_to_nodes.ref_nodes[variants]
    #     variant_nodes = self._variant_to_nodes.var_nodes[variants]
    #     for i, variant in enumerate(self._variants):
    #         if i % 10000 == 0:
    #             print("%d variants genotyped" % i)
    # 
    #         variant_id += 1
    #         self._genotypes_called_at_variant.append(0)
    #         assert "," not in variant.variant_sequence, "Only biallelic variants are allowed. Line is not bialleleic"
    # 
    #         debug = False
    #         reference_node = ref_nodes[i]
    #         variant_node = variant_nodes[i]
    # 
    #         # Compute from actual node counts instead (these are from traversing the graph)
    #         # prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_haplotype_counts(reference_node, variant_node)
    #         prob_homo_ref, prob_homo_alt, prob_hetero = self.get_allele_frequencies_from_most_similar_previous_variant(variant_id)
    # 
    #         predicted_genotype = self._genotype_biallelic_variant(reference_node, variant_node, prob_homo_ref, prob_homo_alt,
    #                                                               prob_hetero, debug)
    #         #self.add_individuals_with_genotype(predicted_genotype, reference_node, variant_node)
    # 
    #         #print("%s\t%s" % ("\t".join(l[0:9]), predicted_genotype))
    #         variant.set_genotype(predicted_genotype)
    # 
    #         numeric_genotype = 0
    #         if predicted_genotype == "0/0":
    #             numeric_genotype = 1
    #         elif predicted_genotype == "1/1":
    #             numeric_genotype = 2
    #         elif predicted_genotype == "0/1":
    #             numeric_genotype = 3
    # 
    #         self._genotypes_called_at_variant[variant_id] = numeric_genotype
