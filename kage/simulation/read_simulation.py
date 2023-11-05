import itertools
import logging

import bionumpy as bnp
import numpy as np
import npstructures as nps
from bionumpy.datatypes import SequenceEntryWithQuality
from kage.util import zip_sequences

from ..indexing.graph import make_multiallelic_graph
from ..indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
from ..preprocessing.variants import get_padded_variants_from_vcf, VariantStreamWithoutVariantsWithSymbolicAlleles


def get_haplotype_genomes(individual_vcf, reference_fasta):
    """Yields two GenomicSequences object (one for each haplotype) containing the sequences of the haplotypes."""
    reference_sequences = bnp.open(reference_fasta).read()
    variant_stream = VariantStreamWithoutVariantsWithSymbolicAlleles.from_vcf(individual_vcf)
    variants, vcf_variants, n_alleles_per_original_variant = get_padded_variants_from_vcf(variant_stream,
                                                                                          reference_fasta,
                                                                                          True,
                                                                                          remove_indel_padding=False)
    graph, node_mapping = make_multiallelic_graph(reference_sequences, variants)

    logging.info("Max n_alleles: %d" % np.max(n_alleles_per_original_variant))
    variant_stream = VariantStreamWithoutVariantsWithSymbolicAlleles.from_vcf(individual_vcf, buffer_type=bnp.io.vcf_buffers.PhasedHaplotypeVCFMatrixBuffer)
    haplotype_matrix_original_vcf = SparseHaplotypeMatrix.from_vcf(variant_stream, dtype=np.uint16)
    biallelic_haplotype_matrix = haplotype_matrix_original_vcf.to_biallelic(n_alleles_per_original_variant)
    n_alleles_per_variant = node_mapping.n_alleles_per_variant
    haplotype_matrix = biallelic_haplotype_matrix.to_multiallelic(n_alleles_per_variant)

    haplotype0 = haplotype_matrix.get_haplotype(0)
    haplotype1 = haplotype_matrix.get_haplotype(1)

    # Use ACGTn-encoding because some variants may have N in them and we want to support that
    sequence1 = graph.sequence(haplotype0, encoding=bnp.encodings.ACGTnEncoding).ravel()
    sequence2 = graph.sequence(haplotype1, encoding=bnp.encodings.ACGTnEncoding).ravel()

    return sequence1, sequence2


def add_errors_to_sequences(sequences: bnp.EncodedRaggedArray,
                            snp_error_rate: float = 0.001, rng=np.random.default_rng()):
    flat_sequences = sequences.ravel()
    effective_mutation_rate = snp_error_rate * 1.25  # scale up since 1/4 of mutations will be to same base
    mutation_locations = rng.integers(0, len(flat_sequences), size=int(len(flat_sequences) * effective_mutation_rate))
    new_bases = rng.integers(0, 4, size=len(mutation_locations))
    new_bases = bnp.EncodedArray(new_bases, bnp.DNAEncoding)
    flat_sequences[mutation_locations] = new_bases


def simulate_reads_from_sequence(sequence: bnp.EncodedArray, read_length: int = 150,
                                 n_reads: int = 10000, snp_error_rate=0.001, rng=np.random.default_rng()):
    starts = rng.integers(0, len(sequence) - read_length, size=n_reads)
    stops = starts + read_length
    sequences = bnp.ragged_slice(sequence, starts, stops)
    add_errors_to_sequences(sequences, rng=rng, snp_error_rate=snp_error_rate)

    # remove sequences with N
    #sequences = sequences[~np.any(sequences == "N", axis=1)]
    #sequences = bnp.change_encoding(sequences, bnp.DNAEncoding)
    revcomps = bnp.sequence.get_reverse_complement(sequences)

    should_be_revcomp = rng.integers(0, 2, size=len(sequences)).astype(bool)
    sequences[should_be_revcomp] = revcomps[should_be_revcomp]

    return sequences


def simulate_paired_end_reads_from_sequence(sequence: bnp.EncodedArray, read_length: int = 150,
                                            n_pairs: int = 1000, snp_error_rate=0.001, rng=np.random.default_rng(),
                                            insert_size: int = 500, insert_size_std: int = 50):
    insert_sizes = rng.normal(insert_size, insert_size_std, size=n_pairs)
    logging.info("Insert sizes: %s" % insert_sizes)
    fragment_starts = rng.integers(0, len(sequence) - insert_sizes, size=n_pairs)
    fragment_ends = fragment_starts + insert_sizes

    reads1 = bnp.ragged_slice(sequence, fragment_starts, fragment_starts + read_length)
    reads2 = bnp.ragged_slice(sequence, fragment_ends - read_length, fragment_ends)[::-1]

    add_errors_to_sequences(reads1, rng=rng, snp_error_rate=snp_error_rate)
    add_errors_to_sequences(reads2, rng=rng, snp_error_rate=snp_error_rate)

    return reads1, reads2


def simulate_reads(individual_vcf: str, reference_fasta: str, out_file_name: str, read_length: int = 150,
                   n_reads: int = 10000, coverage: float = 0, chunk_size: int = 100000, snp_error_rate=0.001,
                   random_seed: int = 1,
                   sequence_name_prefix="",
                   paired_end: bool = False,
                   paired_end_insert_size: int = 500,
                   paired_end_insert_sd: int = 50):
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
                if paired_end:
                    reads1, reads2 = simulate_paired_end_reads_from_sequence(haplotype_seq, read_length, n//2,
                                                                                snp_error_rate, rng,
                                                                                paired_end_insert_size,
                                                                                paired_end_insert_sd)
                    reads = zip_sequences(reads1, reads2, encoding=bnp.encodings.ACGTnEncoding)
                    names = bnp.as_encoded_array(
                        list(itertools.chain.from_iterable(zip(
                            [f"{sequence_name_prefix}{i} 1" for i in range(n_simulated, n_simulated + len(reads1))],
                            [f"{sequence_name_prefix}{i} 2" for i in range(n_simulated, n_simulated + len(reads2))],
                        )))
                    )
                else:
                    reads = simulate_reads_from_sequence(haplotype_seq, read_length, n, snp_error_rate, rng)
                    names = bnp.as_encoded_array(
                        [f"{sequence_name_prefix}{i}" for i in range(n_simulated, n_simulated + len(reads))])

                qualities = nps.RaggedArray(np.ones(reads.size) * 40, reads.shape)
                sequence_entry = SequenceEntryWithQuality(
                    names, reads, qualities)

                f.write(sequence_entry)

                n_simulated += len(sequence_entry)
                logging.info("Simulated %d/%d reads" % (n_simulated, to_simulate))




