import logging
from scipy.special import comb


def get_variant_type(vcf_line):

    l = vcf_line.split()
    if len(l[3]) == len(l[4]):
        return "SNP"
    elif "VT=SNP" in vcf_line:
        return "SNP"
    elif "VT=INDEL" in vcf_line:
        if len(l[3]) > len(l[4]):
            return "DELETION"
        else:
            return "INSERTION"
    else:
        raise Exception("Unsupported variant type on line %s" % vcf_line)


# Genotypes a vcf from node and edge counts in the graph
class CountGenotyper:

    def __init__(self, genotyper, graph, sequence_graph, vcf_file_name, reference_path):
        self.genotyper = genotyper
        self.vcf_file_name = vcf_file_name
        self.graph = graph
        self.sequence_graph = sequence_graph
        self.reference_path = reference_path
        self.expected_read_error_rate = 0.001


    def _store_processed_variant(self, line, edge):
        pass

    def _genotype_biallelic_snp(self, reference_node, variant_node, allele_frequency, debug=False, variant_line=""):
        #logging.info("Genotyping biallelic SNP with nodes %d/%d and allele frequency %.5f" % (reference_node, variant_node, allele_frequency))
        apriori_haplotype_probabilities = [1-allele_frequency, allele_frequency]
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

        a_priori_homozygous_ref = float(variant_line.split("AF_HOMO_REF=")[1].split(";")[0]) + 0.001
        a_priori_homozygous_alt = float(variant_line.split("AF_HOMO_ALT=")[1].split(";")[0]) + 0.001
        a_priori_heterozygous = 1 - a_priori_homozygous_alt - a_priori_homozygous_ref

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

    def genotype(self):

        for i, line in enumerate(open(self.vcf_file_name)):
            if i % 20000 == 0:
                logging.info("%d lines processed" % i)

            if line.startswith("#"):
                if line.startswith("#CHROM"):
                    self._n_haplotypes = (len(line.split()) - 9) * 2
                    logging.info("There are %d haplotypes in this file" % self._n_haplotypes)
                    print(line.strip() + "\tDONOR")
                else:
                    print(line.strip())

                continue

            variant_type = get_variant_type(line)
            l = line.split()
            ref_allele = l[3]
            variant_allele = l[4].lower()
            ref_offset = int(l[1]) - 1
            assert "," not in variant_allele, "Only biallelic variants are allowed. Line is not bialleleic"

            allele_frequency = float(line.split("AF=")[1].split(";")[0])

            # Never accept allele freq 1.0, there is a chance this sample is different
            if allele_frequency == 1.0:
                allele_frequency = 0.99

            if variant_type == "SNP":
                debug = False
                if ref_offset == 1130902-1:
                    debug = True
                reference_node, variant_node = self._process_substitution(ref_offset, variant_allele)
                predicted_genotype = self._genotype_biallelic_snp(reference_node, variant_node, allele_frequency, debug, line)
            elif variant_type == "DELETION":
                continue
                #edge = self._process_deletion(ref_offset+1, len(variant_allele)-1)
            else:
                continue

            print("%s\t%s" % ("\t".join(l[0:9]), predicted_genotype))

    def _process_substitution(self, ref_offset, variant_bases):
        node = self.reference_path.get_node_at_offset(ref_offset)
        node_offset = self.reference_path.get_node_offset_at_offset(ref_offset)
        assert node_offset == 0
        prev_node = self.reference_path.get_node_at_offset(ref_offset - 1)

        # Try to find next node that matches read base
        for potential_next in self.graph.adj_list[prev_node]:
            if potential_next == node:
                continue
            node_seq = self.sequence_graph.get_sequence(potential_next, 0, 1)
            if node_seq.lower() == variant_bases.lower():
                return node, potential_next

        logging.error("Could not parse substitution at offset %d with bases %s" % (ref_offset, variant_bases))
        raise Exception("Parseerrror")

    def _process_deletion(self, ref_offset, deletion_length):
        logging.info("Processing deletion at ref pos %d with size %d" % (ref_offset, deletion_length))
        node = self.reference_path.get_node_at_offset(ref_offset)
        node_offset = self.reference_path.get_node_offset_at_offset(ref_offset)

        print("Processing deltion %s, %d, node offset %d" % (ref_offset, deletion_length, node_offset))
        assert node_offset == 0

        prev_node = self.reference_path.get_node_at_offset(ref_offset - 1)

        # Find next reference node with offset corresponding to the number of deleted base pairs
        next_ref_pos = ref_offset + deletion_length
        next_ref_node = self.reference_path.get_node_at_offset(ref_offset + deletion_length)
        if self.reference_path.get_node_offset_at_offset(next_ref_pos) != 0:
            logging.error("Offset %d is not at beginning of node" % next_ref_pos)
            logging.error("Node at %d: %d" % (next_ref_pos, next_ref_node))
            logging.error("Ref length in deletion: %s" % deletion_length)
            logging.info("Ref pos beginning of deletion: %d" % ref_offset)
            raise Exception("Deletion not in graph")

        self.n_deletions += 1
        return (prev_node, next_ref_node)

    def _process_insertion(self, ref_offset, read_offset):
        base = self.bam_entry.query_sequence[read_offset]
        node = self.reference_path.get_node_at_offset(ref_offset)
        node_offset = self.reference_path.get_node_offset_at_offset(ref_offset)
        node_size = self.graph.blocks[node].length()
        if node_offset != node_size - 1:
            # We are not at end of node, insertion is not represented in the graph, ignore
            return False

        # Find out which next node matches the insertion
        for potential_next in self.graph.adj_list[node]:
            if potential_next in self.linear_reference_nodes:
                continue  # Next should not be in linear ref

            #print("Processing insertion at ref offset %d with base %s" % (ref_offset, base))
            #print("  Node %d with offset %d. Node size: %d" % (node, node_offset, self.graph.blocks[node].length()))

            next_base = self.sequence_graph.get_sequence(potential_next, 0, 1).upper()
            if next_base == base.upper():
                self._variant_edges_detected.add((node, potential_next))
                self.n_insertions += 1
                #print("  Found next node %d with seq %s" % (potential_next, next_base))
                return

