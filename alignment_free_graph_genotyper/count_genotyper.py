import logging
from scipy.special import comb
from .variants import VariantGenotype
from collections import defaultdict
import numpy as np

def parse_vcf_genotype(genotype):
    return genotype.replace("|", "/").replace("1/0", "1/0")


# Genotypes a vcf from node and edge counts in the graph
class CountGenotyper:

    def __init__(self, genotyper, graph, sequence_graph, vcf_file_name, reference_path, variant_window_size=500, n_individuals=2600):
        self.genotyper = genotyper
        self.vcf_file_name = vcf_file_name
        self.graph = graph
        self.sequence_graph = sequence_graph
        self.reference_path = reference_path
        self.expected_read_error_rate = 0.01  # 0.001
        self.n_deletions = 0
        self._variant_window_size = variant_window_size
        self._individuals_with_genotypes = []
        self._individual_counts = np.zeros((variant_window_size, n_individuals))
        self._variant_counter = 0

    def _store_processed_variant(self, line, edge):
        pass

    def add_individuals_with_genotype(self, vcf_line, genotype):
        #individuals = []
        for i, vcf_column in enumerate(vcf_line[9:]):
            individual_genotype = parse_vcf_genotype(vcf_column[0:3])
            individual_id = i
            if genotype == individual_genotype:
                #individuals.append(individual_id)
                self._individual_counts[self._variant_counter % self._variant_window_size][individual_id] = 1
            else:
                self._individual_counts[self._variant_counter % self._variant_window_size][individual_id] = 0

        #self._individuals_with_genotypes.append(individuals)

    def get_most_common_individual_on_previous_variants(self):
        return np.argmax(np.sum(self._individual_counts, 0))

        counts = defaultdict(int)
        for i in range(0, n_variants):
            if i >= len(self._individuals_with_genotypes):
                continue
            for individual in self._individuals_with_genotypes[-i]:
                counts[individual] += 1

        if len(counts) == 0:
            return False

        most_common = max(counts, key=counts.get)
        return most_common

    def get_genotype_for_individual(self, vcf_line, individual_id):
        entry = vcf_line[9:][individual_id]
        return parse_vcf_genotype(entry)

    def _genotype_biallelic_snp(self, reference_node, variant_node, a_priori_homozygous_ref, a_priori_homozygous_alt, a_priori_heterozygous, debug=False):
        #logging.info("Genotyping biallelic SNP with nodes %d/%d and allele frequency %.5f" % (reference_node, variant_node, allele_frequency))
        allele_counts = [self.genotyper.get_node_count(reference_node), self.genotyper.get_node_count(variant_node)]


        tot_counts = sum(allele_counts)
        e = self.expected_read_error_rate
        # Simple when we have bialleleic. Formula for multiallelic given in malva supplmentary
        p_counts_given_homozygous_alt = comb(tot_counts, allele_counts[1]) * (1-e)**allele_counts[1] * e**(tot_counts-allele_counts[1])
        p_counts_given_homozygous_ref = comb(tot_counts, allele_counts[0]) * (1-e)**allele_counts[0] * e**(tot_counts-allele_counts[0])
        #p_counts_given_heterozygous = 1 - p_counts_given_homozygous_alt - p_counts_given_homozygous_ref
        p_counts_given_heterozygous = 1 * comb(tot_counts, allele_counts[0]) * ((1-e)/2)**allele_counts[0] * ((1-e)/2)**allele_counts[1]

        #a_priori_homozygous_ref = (1-allele_frequency)**2
        #a_priori_homozygous_alt = allele_frequency**2
        #a_priori_homozygous_ref = (1-allele_frequency) * 0.9
        #a_priori_homozygous_alt = allele_frequency * 0.9
        #a_priori_heterozygous = 1 - a_priori_homozygous_alt - a_priori_homozygous_ref

        #a_priori_homozygous_ref = float(variant_line.split("AF_HOMO_REF=")[1].split(";")[0]) + 0.001
        #a_priori_homozygous_alt = float(variant_line.split("AF_HOMO_ALT=")[1].split(";")[0]) + 0.001
        #a_priori_heterozygous = 1 - a_priori_homozygous_alt - a_priori_homozygous_ref

        # Denominator in bayes formula
        prob_counts = a_priori_homozygous_ref * p_counts_given_homozygous_ref + \
                        a_priori_homozygous_alt * p_counts_given_homozygous_alt + \
                        a_priori_heterozygous * p_counts_given_heterozygous


        prob_posteriori_heterozygous = a_priori_heterozygous * p_counts_given_heterozygous / prob_counts
        prob_posteriori_homozygous_alt = a_priori_homozygous_alt * p_counts_given_homozygous_alt / prob_counts
        prob_posteriori_homozygous_ref = a_priori_homozygous_ref * p_counts_given_homozygous_ref / prob_counts


        if debug:
            logging.info("==== Nodes: %d / %d" % (reference_node, variant_node))
            logging.info("Alle counts: %s" % allele_counts)
            logging.info("Prob counts given 00, 11, 01: %.5f, %.5f, %.10f" % (p_counts_given_homozygous_ref, p_counts_given_homozygous_alt, p_counts_given_heterozygous))
            logging.info("A priori probs: 00, 11, 01: %.3f, %.3f, %.3f" % (a_priori_homozygous_ref, a_priori_homozygous_alt, a_priori_heterozygous))
            logging.info("Prob of counts: %.3f" % prob_counts)
            logging.info("Posteriori probs for 00, 11, 01: %.4f, %.4f, %.4f" % (prob_posteriori_homozygous_ref, prob_posteriori_homozygous_alt, prob_posteriori_heterozygous))

        # Minimum counts for genotyping
        if allele_counts[1] < 0:
            return "0/0"

        if prob_posteriori_homozygous_ref > prob_posteriori_homozygous_alt and prob_posteriori_homozygous_ref > prob_posteriori_heterozygous:
            return "0/0"
        elif prob_posteriori_homozygous_alt > prob_posteriori_heterozygous:
            return "1/1"
        else:
            return "0/1"

    def compute_a_priori_probabilities(self, genotype, vcf_line):
        most_likely_individual = self.get_most_common_individual_on_previous_variants()
        #logging.info("Most likely individual: %d" % most_likely_individual)
        if most_likely_individual:
            most_likely_individual_genotype = self.get_genotype_for_individual(vcf_line, most_likely_individual)

        prob_following_same_individual = 0.95
        prob_breaking = 1 - prob_following_same_individual

        if False and most_likely_individual:
            logging.info("Most common individual is %d and has genotype %s" % (most_likely_individual, most_likely_individual_genotype))
        if most_likely_individual and most_likely_individual_genotype == genotype:
            return 0.95
        else:
            # Get population probs, there's 0.025 chance for having these
            try:
                a_priori_homozygous_ref = float(vcf_line[7].split("AF_HOMO_REF=")[1].split(";")[0]) + 0.001
                a_priori_homozygous_alt = float(vcf_line[7].split("AF_HOMO_ALT=")[1].split(";")[0]) + 0.001
                a_priori_heterozygous = 1 - a_priori_homozygous_alt - a_priori_homozygous_ref
            except IndexError:
                logging.error("Could not find AF_HOMO_REF/ALT tags in vcf. Info column is: %s" % vcf_line[6])
                raise



            if genotype == "0/0":
                return prob_breaking * a_priori_homozygous_ref
            elif genotype == "0/1":
                return prob_breaking * a_priori_heterozygous
            elif genotype == "1/1":
                return prob_breaking * a_priori_homozygous_alt
            else:
                raise Exception("Invalid genotype %s" % genotype)


    def genotype(self):

        for i, line in enumerate(open(self.vcf_file_name)):
            if i % 100 == 0:
                logging.info("%d lines processed" % i)

            if line.startswith("#"):
                if line.startswith("#CHROM"):
                    self._n_haplotypes = (len(line.split()) - 9) * 2
                    logging.info("There are %d haplotypes in this file" % self._n_haplotypes)
                    print("\t".join(line.split()[0:9]).strip() + "\tDONOR")
                else:
                    print(line.strip())

                continue

            l = line.split()
            variant = VariantGenotype.from_vcf_line(line)
            assert "," not in variant.variant_sequence, "Only biallelic variants are allowed. Line is not bialleleic"

            prob_homo_ref = self.compute_a_priori_probabilities("0/0", l)
            prob_homo_alt = self.compute_a_priori_probabilities("1/1", l)
            prob_hetero = self.compute_a_priori_probabilities("0/1", l)

            debug = False
            reference_node, variant_node = self.graph.get_variant_nodes(variant)
            predicted_genotype = self._genotype_biallelic_snp(reference_node, variant_node, prob_homo_ref, prob_homo_alt, prob_hetero, debug)

            self.add_individuals_with_genotype(l, predicted_genotype)
            if len(self._individuals_with_genotypes) > 40:
                self._individuals_with_genotypes.pop(0)

            self._variant_counter += 1

            print("%s\t%s" % ("\t".join(l[0:9]), predicted_genotype))
