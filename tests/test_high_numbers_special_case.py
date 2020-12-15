import numpy as np
from graph_kmer_index import kmer_hash_to_sequence

hashes = np.array([3728905935996505907], dtype=np.uint64)
h = hashes[0]

print(kmer_hash_to_sequence(h, 31))