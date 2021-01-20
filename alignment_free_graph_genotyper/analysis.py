from .variants import get_variant_type
import logging
from collections import defaultdict
from .variants import TruthRegions, VariantGenotype, GenotypeCalls
from graph_kmer_index import kmer_hash_to_sequence, sequence_to_kmer_hash

from obgraph import VariantNotFoundException

class KmerAnalyser:
    def __init__(self, graph, k, variants, kmer_index, reverse_kmer_index, predicted_genotypes, truth_genotypes, truth_regions, node_counts, node_count_model, genotype_frequencies, most_similar_variants):
        self.graph = graph
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
        self.node_counts = node_counts
        self.genotype_frequencies = genotype_frequencies
        self.most_similar_variants = most_similar_variants

    def print_report(self):
        for type in ["SNP", "DELETION", "INSERTION"]:
            print("Type: %s" % type)
            print("N truth variants: %d" % self.n_truth_variants[type])
            print("N predicted variants: %d" % self.n_predicted_variants[type])
            print("N correctly predicted: %d" % self.n_correct_genotypes[type])
            print("N false postives: %d" % self.false_positives[type])

    def print_info_about_variant(self, reference_node, variant_node, variant, variant_id):
        reference_kmers = set(self.reverse_kmer_index.get_node_kmers(reference_node))
        variant_kmers = set(self.reverse_kmer_index.get_node_kmers(variant_node))
        logging.warning("Predicted: %s. Variant id: %d" % (self.predicted_genotypes.get(variant), variant_id))
        logging.warning("Node counts on %d/%d:             %d/%d" % (
        reference_node, variant_node, self.node_counts.node_counts[reference_node],
        self.node_counts.node_counts[variant_node]))
        logging.warning("Model counts following nodes:     %.3f/%.3f" % (
        self.node_count_model.node_counts_following_node[reference_node],
        self.node_count_model.node_counts_following_node[variant_node]))
        logging.warning("Model counts not following nodes: %.3f/%.3f" % (
        self.node_count_model.node_counts_not_following_node[reference_node],
        self.node_count_model.node_counts_not_following_node[variant_node]))
        most_similar = self.most_similar_variants.get_most_similar_variant(variant_id)
        logging.warning("Most similar to variant %d with similarity %.4f and call %s" % (most_similar, self.most_similar_variants.prob_of_having_the_same_genotype_as_most_similar(variant_id), self.predicted_genotypes[most_similar]))
        logging.warning("Genotype frequencies: %.3f/%.3f/%.3f" % self.genotype_frequencies.get_frequencies_for_variant(variant_id))
        logging.warning("Most similar frequencies: %.3f/%.3f/%.3f" % self.genotype_frequencies.get_frequencies_for_variant(most_similar))
        logging.warning("Kmers on ref/variant node %d/%d: %s / %s" % (reference_node, variant_node, reference_kmers, variant_kmers))
        logging.warning("Kmer sequences on variant node:   %s" % [(hash, kmer_hash_to_sequence(hash, 31)) for hash in variant_kmers])
        logging.warning("Kmer sequences on reference node: %s" % [(hash, kmer_hash_to_sequence(hash, 31)) for hash in reference_kmers])
        #logging.warning("Kmer sequences on ref     node: %s" % [kmer_hash_to_sequence(hash, 31) for hash in reference_kmers])
        if variant.type == "SNP":
            prev_node_difference = 2
        elif variant.type == "DELETION" or variant.type == "INSERTION":
            prev_node_difference = 1
        prev_node = self.graph.get_node_at_ref_offset(variant.position - prev_node_difference)
        next_node = self.graph.get_node_at_ref_offset(
            variant.position + self.graph.get_node_size(reference_node) + prev_node_difference)
        logging.warning("Ref nodes in area: %s" % str(
            [self.graph.get_node_sequence(node) for node in [prev_node, reference_node, next_node]]))

    def analyse_variant(self, reference_node, variant_node, variant, variant_id):
        reference_kmers = set(self.reverse_kmer_index.get_node_kmers(reference_node))
        variant_kmers = set(self.reverse_kmer_index.get_node_kmers(variant_node))

        if self.truth_genotypes.has_variant(variant):
            self.n_truth_variants[variant.type] += 1

        if self.predicted_genotypes.has_variant(variant):
            self.n_predicted_variants[variant.type] += 1

            if self.truth_genotypes.has_variant(variant):
                if self.truth_genotypes.get(variant) == self.predicted_genotypes.get(variant):
                    self.n_correct_genotypes[variant.type] += 1
                #else:
                #logging.warning("Wrong genotype: %s / %s" % (self.truth_genotypes.get(variant), self.predicted_genotypes.get(variant)))

        if False and variant.type == "SNP" and self.predicted_genotypes.has_variant(variant) and self.truth_genotypes.has_variant(variant):
            if self.truth_regions.is_inside_regions(variant.position) and self.predicted_genotypes.get(variant).genotype == "0|0" and self.truth_genotypes.get(variant).genotype != "0|0":
                logging.warning("----------------------------")
                logging.warning("False negative genotype!")
                self.print_info_about_variant(reference_node, variant_node, variant)

        if variant.type == "DELETION" and self.predicted_genotypes.has_variant(variant) and not self.truth_genotypes.has_variant(variant):
            if self.truth_regions.is_inside_regions(variant.position) and self.predicted_genotypes.get(variant).genotype != "0|0":
                logging.warning("False positive genotype!")
                self.print_info_about_variant(reference_node, variant_node, variant, variant_id)

                self.false_positives[variant.type] += 1


        if len(reference_kmers.intersection(variant_kmers)) > 0 and False:
            logging.warning("----------")
            logging.warning("Variant and reference node share kmers on %d/%d: %s / %s" % (reference_node, variant_node, reference_kmers, variant_kmers))
            # Check if they have any of the same kmers
            if len(reference_kmers) == 0:
                logging.warning("Variant %s %d/%d has 0 reference kmers" % (variant, reference_node, variant_node))
            if len(reference_kmers) == 0:
                logging.warning("Variant %s %d/%d has 0 variant kmers" % (variant, reference_node, variant_node))

            #if variant.type == "DELETION" or variant.type == "INSERTION":
            if variant.type == "SNP":
                logging.warning("Reference node sequence: %s" % self.graph.get_node_sequence(reference_node))
                logging.warning("Next reference node sequence: %s" % self.graph.get_node_sequence(self.graph.get_edges(reference_node)[0]))

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
                if variant.type == "SNP":
                    reference_node, variant_node = self.graph.get_snp_nodes(ref_offset, variant_allele)
                elif variant.type == "DELETION":
                    self.n_deletions += 1
                    reference_node, variant_node = self.graph.get_deletion_nodes(ref_offset, len(ref_allele)-1)
                elif variant.type == "INSERTION":
                    self.n_insertions += 1
                    reference_node, variant_node = self.graph.get_variant_nodes(variant)

                else:
                    continue
            except VariantNotFoundException:
                continue

            self.analyse_variant(reference_node, variant_node, variant, i)

        logging.info("N insertions: %d" % self.n_insertions)
        logging.info("N deletions: %d" % self.n_deletions)
        logging.info("N ambiguous deletions/insertions: %d" % self.n_ambiguous)
        logging.info("N ambiguous deletions wrongly genotyped: %d" % self.n_ambiguous_wrongly_genotyped)
        print(self.n_truth_variants)
        self.print_report()
