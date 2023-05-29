from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix
import numpy as np


def test_from_variants_and_haplotypes():
    variant_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    haplotype_ids = np.array([0, 1, 2, 1, 2, 0, 1, 2])
    matrix = SparseHaplotypeMatrix.from_variants_and_haplotypes(variant_ids, haplotype_ids, 3, 3)
    assert matrix.data.toarray().tolist() == [
        [1, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    assert matrix.data.shape == (3, 3)
    print(matrix.data.toarray())
    assert np.all(matrix.data.getrow(0).toarray() == np.array([1, 1, 1]))

    assert np.all(matrix.get_haplotype(0) == [1, 0, 1])