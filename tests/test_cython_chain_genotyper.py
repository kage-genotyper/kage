import pyximport; pyximport.install(language_level=3)
from pyfaidx import Fasta
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from graph_kmer_index import KmerIndex
import time
from alignment_free_graph_genotyper.genotyper import ReadKmers

from alignment_free_graph_genotyper.cython_chain_genotyper import run
from numpy_alignments import NumpyAlignments
truth_positions = NumpyAlignments.from_file("truth10k.npz")

index = KmerIndex.from_file("merged31")
logging.info("Done reading index")

fasta_file_name = "simulated_reads10k.fa"
reference_k = 16
reference_kmers = \
    ReadKmers.get_kmers_from_read_dynamic(str(Fasta("ref.fa")["1"]), np.power(4, np.arange(0, reference_k)))
logging.info("Done creating reference kmers")

start_time = time.time()
chain_positions, node_counts = run(fasta_file_name,
            index._hasher._hashes,
            index._hashes_to_index,
            index._n_kmers,
            index._nodes,
            index._ref_offsets,
            reference_kmers,
            10000000
)

end_time = time.time()
logging.info("Time spent: %.5f" % (end_time - start_time))

print(chain_positions)

chain_positions = chain_positions[0:10000]
n_correct = len(np.where(np.abs(chain_positions - truth_positions.positions) <= 150)[0])
logging.info("N correct: %d" % n_correct)

