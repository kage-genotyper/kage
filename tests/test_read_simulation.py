import bionumpy as bnp
from kage.simulation.read_simulation import simulate_reads_from_sequence



def test_simple():
    sequence = bnp.as_encoded_array("ACTGACAACANCACTACCA")
    simulated = simulate_reads_from_sequence(sequence, read_length=3, n_reads=10)

    print(simulated)



def test_get_haplotype_genomes():
    pass

