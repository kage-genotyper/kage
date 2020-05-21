import os
import logging
import numpy as np
from .genotyper import ReadKmers
from pyfaidx import Fasta

class ReferenceKmers:
    def __init__(self, fasta_file_name, reference_name, k, allow_cache=True):
        reference_kmers_cache_file_name = fasta_file_name + ".%dmers" % k
        if os.path.isfile(reference_kmers_cache_file_name + ".npy"):
            logging.info("Used cached reference kmers from file %s.npy" % reference_kmers_cache_file_name)
            self.reference_kmers = np.load(reference_kmers_cache_file_name + ".npy")
        else:
            logging.info("Creating reference kmers")
            self.reference_kmers = ReadKmers.get_kmers_from_read_dynamic(str(Fasta("ref.fa")[reference_name]), np.power(4, np.arange(0, k)))
            np.save(reference_kmers_cache_file_name, self.reference_kmers)

