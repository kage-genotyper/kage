import pyximport; pyximport.install(language_level=3)
from alignment_free_graph_genotyper.chaining import chain, chain_with_score
import numpy as np
from alignment_free_graph_genotyper.chain_genotyper import ChainGenotyper
from alignment_free_graph_genotyper.genotyper import ReadKmers
from alignment_free_graph_genotyper.chain_genotyper import read_kmers
from pyfaidx import Fasta
import time
import sys

power_array = np.power(4, np.arange(0, 16))
read = "GTCTTCCGAGCGTCAGGCCGCCCCTACCCGTGCTTTCTGCTCTGCAGACCCTCTTCCTAGACCTCCGTCCTTTGTCCCATCGCTGCCTTCCCCTCAAGCTCAGGGCCAAGCTGTCCGCCAACCTCGGCTCCTCCGGGCAGCCCTCGCCCG"
kmers = read_kmers(read, power_array)
reference_kmers = \
    ReadKmers.get_kmers_from_read_dynamic(str(Fasta("ref.fa")["1"]), np.power(4, np.arange(0, 16)))

read_offsets = np.array([10, 20, 50, 51, 90, 90, 50, 40, 50, 65, 87], dtype=np.int)
ref_offsets = np.array([10011, 10021, 10099, 100100, 100101, 100200, 1001232, 1003222, 1002121, 100321, 100322], dtype=np.int)
nodes = np.array([1, 2, 50, 52, 4, 100, 101, 202, 103, 104, 110], dtype=np.int)

def test_python():
    chains = ChainGenotyper.find_chains(ref_offsets, read_offsets, nodes)
    #print(chains)

def test_cython():
    chains = chain(ref_offsets, read_offsets, nodes)
    #print(chains)

def test_cython_best_chain():
    best = chain_with_score(ref_offsets, read_offsets, nodes, reference_kmers, kmers)
    #print(best)

def run_test(func):
    start = time.time()
    for i in range(5000):
        func()
    end = time.time()
    print(end - start)

test_python()
test_cython_best_chain()
run_test(test_cython_best_chain)
run_test(test_python)
run_test(test_cython)
sys.exit()

#sys.exit()

