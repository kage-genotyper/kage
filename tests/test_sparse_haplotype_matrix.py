from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix, GenotypeMatrix
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


def test_genotype_matrix():
    haplotype_matrix = SparseHaplotypeMatrix.from_nonsparse_matrix(
        [[1, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 0]]
    )
    genotype_matrix = GenotypeMatrix.from_haplotype_matrix(haplotype_matrix)

    assert np.all(genotype_matrix.matrix == np.array([
        [2, 1],
        [0, 0],
        [1, 1],
    ]))
