from obgraph.variants import get_variant_type
import logging
from collections import defaultdict
from obgraph.variants import TruthRegions, VcfVariant, VcfVariants
from graph_kmer_index import kmer_hash_to_sequence, sequence_to_kmer_hash

from obgraph import VariantNotFoundException

class SimpleRecallPrecisionAnalyser:
    def __init__(self, predicted_variants, truth_variants, truth_regions):
        self.predicted_variants = predicted_variants
        self.truth_variants = truth_variants
        self.truth_regions = truth_regions

        self._n_truth = defaultdict(int)
        self._true_positive = defaultdict(int)
        self._false_negative = defaultdict(int)
        self._false_positive = defaultdict(int)

    def analyse(self):
        for truth in self.truth_variants:
            self._n_truth[truth.type] += 1
            if self.predicted_variants.has_variant(truth):
                if self.predicted_variants.get(truth).get_genotype() == truth.get_genotype():
                    self._true_positive[truth.type] += 1

            if not self.predicted_variants.has_variant(truth) or (self.predicted_variants.has_variant(truth) and self.predicted_variants.get(truth).get_genotype() == "0/0"):
                self._false_negative[truth.type] += 1

        for variant in self.predicted_variants:

            if self.truth_regions.is_inside_regions(variant.position):
                if variant.get_genotype() != "0/0" and not self.truth_variants.has_variant(variant):
                    self._false_positive[variant.type] += 1


        logging.info("--- REPORT ---")
        for type in ["SNP", "DELETION", "INSERTION"]:
            precision = self._true_positive[type] / (self._true_positive[type] + self._false_positive[type])
            recall = self._true_positive[type] / (self._true_positive[type] + self._false_negative[type])
            logging.info("%s. Recall: %.4f, Precision: %.4f" % (type, recall, precision))


class GenotypeDebugger:
    def __init__(self, variant_nodes, k, variants, kmer_index, reverse_kmer_index, predicted_genotypes, truth_genotypes,
                 truth_regions, node_counts, node_count_model, helper_variants, combination_matrix, probs, count_probs,
                 pangenie):
        self.variant_nodes= variant_nodes
        self.node_count_model = node_count_model
        self.k = k
        self.variants = variants
        self.kmer_index = kmer_index
        self.reverse_kmer_index = reverse_kmer_index
        self.n_ambiguous = 0
        self.n_ambiguous_wrongly_genotyped = 0
        self.n_deletions = 0
        self.n_insertions = 0
        self.predicted_genotypes = predicted_genotypes
        self.truth_regions = truth_regions
        self.truth_genotypes = truth_genotypes
        self.n_truth_variants = defaultdict(int)
        self.n_predicted_variants = defaultdict(int)
        self.n_correct_genotypes = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.false_negatives_with_zero_counts = defaultdict(int)
        self.false_negatives_with_zero_counts_and_other_truth_variant_close = defaultdict(int)
        self.false_negatives_with_zero_counts_and_other_truth_variant_close_only_one_side = defaultdict(int)
        self.node_counts = node_counts
        #self.genotype_frequencies = genotype_frequencies
        #self.most_similar_variants = most_similar_variants
        self.n_missing_kmers = 0
        self.n_ref_nodes_zero_in_model = 0
        self.truth_genotypes.make_position_index()
        #self.whitelist = whitelist
        #self.transition_probs = transition_probs
        self.helper_variants = helper_variants
        self.combination_matrix = combination_matrix
        self.probs = probs
        self.count_probs = count_probs
        self.pangenie = pangenie

    def print_report(self):
        for type in ["SNP", "DELETION", "INSERTION"]:
            logging.info("Type: %s" % type)
            logging.info("N truth variants: %d" % self.n_truth_variants[type])
            logging.info("N predicted variants: %d" % self.n_predicted_variants[type])
            logging.info("N correctly predicted: %d" % self.n_correct_genotypes[type])
            logging.info("N false postives: %d" % self.false_positives[type])
            logging.info("N false negatives: %d" % self.false_negatives[type])
            logging.info("N false negatives with zero counts: %d" % self.false_negatives_with_zero_counts[type])
            logging.info("N false negatives with zero counts and truth variant close: %d (only one side: %d)" % (self.false_negatives_with_zero_counts_and_other_truth_variant_close[type], self.false_negatives_with_zero_counts_and_other_truth_variant_close_only_one_side[type]))
            #logging.info("Recall: %.4f" % (self.n_correct_genotypes[type] / self.n_truth_variants[type]))
            logging.info("Precision (all genotype types): %.4f" % (self.n_correct_genotypes[type] / self.n_predicted_variants[type]))

    def print_info_about_variant(self, reference_node, variant_node, variant, variant_id):
        reference_kmers = set(self.reverse_kmer_index.get_node_kmers(reference_node))
        variant_kmers = set(self.reverse_kmer_index.get_node_kmers(variant_node))

        predicted_genotype = self.predicted_genotypes.get(variant).genotype

        if predicted_genotype == "0|0" and not self.pangenie.has_variant(variant):
            # ignore
            return
        elif predicted_genotype != "0|0" and self.pangenie.has_variant(variant) and self.pangenie.get(variant).genotype == predicted_genotype:
            return



        logging.warning("")
        logging.warning("")
        logging.warning("")
        logging.warning("")
        assert self.predicted_genotypes.get(variant).genotype != "", self.predicted_genotypes.get(variant).vcf_line
        logging.warning("Predicted: %s. Correct is %s. Variant id: %d" % (self.predicted_genotypes.get(variant), self.truth_genotypes.get(variant), variant_id))

        if(self.pangenie.has_variant(variant)):
            logging.info("PANGENIE: %s" % self.pangenie.get(variant))

        logging.warning("Node counts on ref/alt %d/%d: %d/%d" % (
        reference_node, variant_node, self.node_counts.node_counts[reference_node],
        self.node_counts.node_counts[variant_node]))

        logging.info("Genotype probs (log): %s" % self.probs[variant_id])
        logging.info("Count    probs (log): %s" % self.count_probs[variant_id])

        logging.info("Model on ref node: %s" % self.node_count_model.describe_node(reference_node))
        logging.info("Model on alt node: %s" % self.node_count_model.describe_node(variant_node))


        most_similar = self.helper_variants[variant_id]
        most_similar_variant = self.predicted_genotypes[most_similar]
        logging.info("Most similar variant: %d" % most_similar)
        logging.info("Predicted/true most similar: %s / %s" % (self.predicted_genotypes[most_similar], self.truth_genotypes.get(most_similar_variant)))
        helper_ref = self.variant_nodes.ref_nodes[most_similar]
        helper_alt = self.variant_nodes.var_nodes[most_similar]
        helper_count_ref = self.node_counts[helper_ref]
        helper_count_alt = self.node_counts[helper_alt]
        logging.info("Node counts on helper nodes %d/%d: %d/%d" % (helper_ref, helper_alt, helper_count_ref, helper_count_alt))
        logging.info("Model on helper ref: %s" % self.node_count_model.describe_node(helper_ref))
        logging.info("Model on helper alt: %s" % self.node_count_model.describe_node(helper_alt))
        logging.info("Combination matrix: \n%s" % self.combination_matrix[variant_id])
        logging.info("Genotype probs most similar (log): %s" % self.probs[most_similar])
        logging.info("Count    probs most similar (log): %s" % self.count_probs[most_similar])

        #logging.info("Model counts ref node (homo ref, homo alt, hetero): %.2f/%.2f/%.2f" % (self.node_count_model.counts_homo_ref[reference_node], self.node_count_model.counts_homo_alt[reference_node], self.node_count_model.counts_hetero[reference_node]))
        #logging.info("Model counts alt node (homo ref, homo alt, hetero): %.2f/%.2f/%.2f" % (self.node_count_model.counts_homo_ref[variant_node], self.node_count_model.counts_homo_alt[variant_node], self.node_count_model.counts_hetero[variant_node]))

        #logging.warning("Model counts following nodes:     %.3f/%.3f" % (
        #self.node_count_model.node_counts_following_node[reference_node],
        #self.node_count_model.node_counts_following_node[variant_node]))
        #logging.warning("Model counts not following nodes: %.3f/%.3f" % (
        #self.node_count_model.node_counts_not_following_node[reference_node],
        #self.node_count_model.node_counts_not_following_node[variant_node]))
        #most_similar = self.most_similar_variants.get_most_similar_variant(variant_id)
        #logging.warning("Most similar to variant %d with similarity %.4f and call %s" % (most_similar, self.most_similar_variants.prob_of_having_the_same_genotype_as_most_similar(variant_id), self.predicted_genotypes[most_similar]))
        #logging.warning("Genotype frequencies: %.3f/%.3f/%.3f" % self.genotype_frequencies.get_frequencies_for_variant(variant_id))
        #logging.warning("Most similar frequencies: %.3f/%.3f/%.3f" % self.genotype_frequencies.get_frequencies_for_variant(most_similar))
        #logging.warning("Transition probabilities from most similar: %s" % self.transition_probs.get_transition_probabilities(variant_id, self.predicted_genotypes[most_similar].get_numeric_genotype()))
        logging.warning("Kmers on ref/variant node %d/%d: %s / %s" % (reference_node, variant_node, reference_kmers, variant_kmers))
        logging.warning("Kmer sequences on variant node:   %s" % [(hash, kmer_hash_to_sequence(hash, 31)) for hash in variant_kmers])
        logging.warning("Kmer sequences on reference node: %s" % [(hash, kmer_hash_to_sequence(hash, 31)) for hash in reference_kmers])
        #logging.warning("Kmer sequences on ref     node: %s" % [kmer_hash_to_sequence(hash, 31) for hash in reference_kmers])
        if variant.type == "SNP":
            prev_node_difference = 2
        elif variant.type == "DELETION" or variant.type == "INSERTION":
            prev_node_difference = 1

    def analyse_variant(self, reference_node, variant_node, variant, variant_id):

        #if self.whitelist is not None and self.whitelist[variant_id] == 0:
        #    # ignore
        #    return

        if not self.truth_regions.is_inside_regions(variant.position):
            return

        reference_kmers = set(self.reverse_kmer_index.get_node_kmers(reference_node))
        variant_kmers = set(self.reverse_kmer_index.get_node_kmers(variant_node))

        if len(self.reverse_kmer_index.get_node_kmers(reference_node)) == 0 or len(self.reverse_kmer_index.get_node_kmers(variant_node)) == 0:
            self.n_missing_kmers += 1
            logging.warning("Node %d or %d does not have kmers, variant %s! %d missing so far" % (reference_node, variant_node, variant, self.n_missing_kmers))

        if self.truth_genotypes.has_variant(variant):
            self.n_truth_variants[variant.type] += 1

        #logging.info(self.predicted_genotypes.get(variant))
        #logging.info(self.truth_genotypes.get(variant))

        if self.predicted_genotypes.has_variant(variant):
            self.n_predicted_variants[variant.type] += 1

            if self.truth_genotypes.has_variant(variant):
                if self.truth_genotypes.get(variant) == self.predicted_genotypes.get(variant):
                    self.n_correct_genotypes[variant.type] += 1
                else:
                    if variant.type != "SNP":
                        #logging.warning("Wrong genotype: %s / %s" % (
                        #    self.truth_genotypes.get(variant), self.predicted_genotypes.get(variant)))
                        self.print_info_about_variant(reference_node, variant_node, variant, variant_id)

            elif self.predicted_genotypes.get(variant).genotype == "0|0":
                self.n_correct_genotypes[variant.type] += 1

        #if self.node_count_model.node_counts_following_node[reference_node] == 0:
        #    self.n_ref_nodes_zero_in_model += 1

        if variant.type != "SNP" and self.predicted_genotypes.has_variant(variant) and self.truth_genotypes.has_variant(variant):
            if self.truth_regions.is_inside_regions(variant.position) and self.predicted_genotypes.get(variant).genotype == "0|0" and self.truth_genotypes.get(variant).genotype != "0|0" and self.truth_genotypes.get(variant).genotype != self.predicted_genotypes.get(variant).genotype:
                logging.warning("----------------------------")
                logging.warning("False negative genotype!")
                self.print_info_about_variant(reference_node, variant_node, variant, variant_id)
                self.false_negatives[variant.type] += 1
                if self.node_counts[variant_node] == 0:
                    self.false_negatives_with_zero_counts[variant.type] += 1
                    window = 31
                    if self.truth_genotypes.has_variant_left_of_variant(variant, window) or self.truth_genotypes.has_variant_right_of_variant(variant, window):
                        self.false_negatives_with_zero_counts_and_other_truth_variant_close[variant.type] += 1
                        if not self.truth_genotypes.has_variant_left_of_variant(variant, window) or not self.truth_genotypes.has_variant_right_of_variant(variant, window):
                            self.false_negatives_with_zero_counts_and_other_truth_variant_close_only_one_side[variant.type] += 1
                            #self.print_info_about_variant(reference_node, variant_node, variant, variant_id)

        if variant.type != "SNP" and self.predicted_genotypes.has_variant(variant) and (not self.truth_genotypes.has_variant(variant) or (self.truth_genotypes.has_variant(variant) and self.truth_genotypes.get(variant).genotype == "0|0")):

            if self.truth_regions.is_inside_regions(variant.position) and self.predicted_genotypes.get(variant).genotype != "0|0":
                logging.warning("----------------------------")
                logging.warning("False positive genotype!")
                self.print_info_about_variant(reference_node, variant_node, variant, variant_id)

                self.false_positives[variant.type] += 1


        if len(reference_kmers.intersection(variant_kmers)) > 0:
            logging.warning("----------")
            logging.warning("Variant and reference node share kmers on variant %s, nodes %d/%d: %s / %s" % (variant, reference_node, variant_node, reference_kmers, variant_kmers))
            # Check if they have any of the same kmers
            if len(reference_kmers) == 0:
                logging.warning("Variant %s %d/%d has 0 reference kmers" % (variant, reference_node, variant_node))
            if len(reference_kmers) == 0:
                logging.warning("Variant %s %d/%d has 0 variant kmers" % (variant, reference_node, variant_node))

            if variant.type == "DELETION" or variant.type == "INSERTION":
            #if variant.type == "SNP":

                self.n_ambiguous += 1

                if self.predicted_genotypes.has_variant(variant):
                    logging.warning("Predicted: %s" % self.predicted_genotypes.get(variant))
                    if (not self.truth_genotypes.has_variant(variant) and self.predicted_genotypes.get(variant).genotype != "0|0") \
                            or (self.truth_genotypes.has_variant(variant) and self.predicted_genotypes.get(variant) != self.truth_genotypes.get(variant)):
                        self.n_ambiguous_wrongly_genotyped += 1

                if self.truth_genotypes.has_variant(variant):
                    logging.warning("Truth: %s" % self.predicted_genotypes.get(variant))

        #logging.info("Variant and reference node on %d/%d: %s / %s" % (reference_node, variant_node, reference_kmers, variant_kmers))

    def analyse_unique_kmers_on_variants(self):
        #for i, line in enumerate(open(self.vcf_file_name)):
        for i, variant in enumerate(self.variants.variant_genotypes):
            if i % 10000 == 0:
                logging.info("%d lines processed" % i)

            ref_allele = variant.ref_sequence
            variant_allele = variant.variant_sequence
            ref_offset = variant.position - 1
            assert "," not in variant_allele, "Only biallelic variants are allowed. Line is not bialleleic"

            try:
                #reference_node, variant_node = self.graph.get_variant_nodes(variant)
                reference_node = self.variant_nodes.ref_nodes[variant.vcf_line_number]
                variant_node = self.variant_nodes.var_nodes[variant.vcf_line_number]
                if variant.type == "DELETION":
                    self.n_deletions += 1
                elif variant.type == "INSERTION":
                    self.n_insertions += 1

            except VariantNotFoundException:
                continue

            self.analyse_variant(reference_node, variant_node, variant, i)

        logging.info("Reference nodes that are zero in model: %d" % self.n_ref_nodes_zero_in_model)
        logging.info("N insertions: %d" % self.n_insertions)
        logging.info("N deletions: %d" % self.n_deletions)
        logging.info("N ambiguous deletions/insertions: %d" % self.n_ambiguous)
        logging.info("N ambiguous deletions wrongly genotyped: %d" % self.n_ambiguous_wrongly_genotyped)
        print(self.n_truth_variants)
        self.print_report()
