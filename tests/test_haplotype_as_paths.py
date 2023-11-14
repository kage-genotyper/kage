import numpy as np
from kage.indexing.path_based_count_model import get_haplotypes_as_paths_parallel
from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix


def test_real_case():
    paths = np.load("path_alleles1.npy")
    haplotype_matrix = SparseHaplotypeMatrix.from_nonsparse_matrix(np.load("haplotype_matrix1.npy"))

    haplotype_as_paths = get_haplotypes_as_paths_parallel(haplotype_matrix, paths,
                                                          window_size=3,
                                                          n_threads=1)

    # All haplotypes should have gotten a match
