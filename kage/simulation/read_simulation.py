import logging

import bionumpy as bnp
import numpy as np
import npstructures as nps
from bionumpy.datatypes import SequenceEntryWithQuality

from ..indexing.graph import make_multiallelic_graph
from ..indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from ..preprocessing.variants import get_padded_variants_from_vcf


def get_haplotype_genomes(individual_vcf, reference_fasta):
    """Yields two GenomicSequences object (one for each haplotype) containing the sequences of the haplotypes."""
    reference_sequences = bnp.open(reference_fasta).read()
    variants, vcf_variants, n_alleles_per_original_variant = get_padded_variants_from_vcf(individual_vcf,
                                                                                          reference_fasta, True)
    graph, node_mapping = make_multiallelic_graph(reference_sequences, variants)

    haplotype_matrix_original_vcf = SparseHaplotypeMatrix.from_vcf(individual_vcf)
    biallelic_haplotype_matrix = haplotype_matrix_original_vcf.to_biallelic(n_alleles_per_original_variant)
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    haplotype_matrix = biallelic_haplotype_matrix.to_multiallelic(n_alleles_per_variant)

    haplotype0 = haplotype_matrix.get_haplotype(0)
    haplotype1 = haplotype_matrix.get_haplotype(1)

    sequence1 = graph.sequence(haplotype0).ravel()
    sequence2 = graph.sequence(haplotype1).ravel()

    return sequence1, sequence2


def simulate_reads_from_sequence(sequence: bnp.EncodedArray, read_length: int = 150, n_reads: int = 10000, rng=np.random.default_rng()):
    starts = rng.integers(0, len(sequence) - read_length, size=n_reads)
    stops = starts + read_length
    sequences = bnp.ragged_slice(sequence, starts, stops)

    # remove sequences with N
    #sequences = sequences[~np.any(sequences == "N", axis=1)]
    #sequences = bnp.change_encoding(sequences, bnp.DNAEncoding)
    revcomps = bnp.sequence.get_reverse_complement(sequences)

    should_be_revcomp = rng.integers(0, 2, size=len(sequences)).astype(bool)
    sequences[should_be_revcomp] = revcomps[should_be_revcomp]

    return sequences


def simulate_reads(individual_vcf: str, reference_fasta: str, out_file_name: str, read_length: int = 150,
                  n_reads: int = 10000, coverage: float = 0, chunk_size: int = 100000, random_seed: int = 1,
                   sequence_name_prefix=""):
    haplotype0, haplotype1 = get_haplotype_genomes(individual_vcf, reference_fasta)

    if coverage > 0:
        n_reads = int(coverage * len(haplotype0) / read_length)
        genome_size = len(haplotype0.ravel())
        logging.info("Genome size: %d" % len(haplotype0))
        logging.info("Setting n reads to be %d to get coverage %.3f" % (n_reads, coverage))

    rng = np.random.default_rng(random_seed)

    with bnp.open(out_file_name, "w") as f:
        for haplotype_seq in (haplotype0, haplotype1):
            to_simulate = n_reads // 2
            n_simulated = 0
            while n_simulated < to_simulate:
                n = min(n_reads - n_simulated, chunk_size)
                reads = simulate_reads_from_sequence(haplotype_seq, read_length, n, rng)

                names = bnp.as_encoded_array(
                    [f"{sequence_name_prefix}{i}" for i in range(n_simulated, n_simulated + len(reads))])
                qualities = nps.RaggedArray(np.ones(reads.size) * 40, reads.shape)
                sequence_entry = SequenceEntryWithQuality(
                    names, reads, qualities)

                f.write(sequence_entry)

                n_simulated += len(sequence_entry)
                logging.info("Simulated %d/%d reads" % (n_simulated, to_simulate))



