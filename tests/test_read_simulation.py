import bionumpy as bnp
import numpy as np
from kage.simulation.read_simulation import simulate_reads_from_sequence, add_errors_to_sequences


def test_simple():
    sequence = bnp.as_encoded_array("ACTGACAACACACTACCA", bnp.DNAEncoding)
    simulated = simulate_reads_from_sequence(sequence, read_length=3, n_reads=10)
    print(simulated)


def test_get_haplotype_genomes():
    pass


def test_add_errors_to_sequences():
    sequences = bnp.as_encoded_array(["ACTGAC", "AAAAAAAAAAAAAaA", "CCCCCCCCCCCCCCCCCCCC"], bnp.DNAEncoding)
    add_errors_to_sequences(sequences, snp_error_rate=0.3, rng=np.random.default_rng(1))
    print(sequences)
