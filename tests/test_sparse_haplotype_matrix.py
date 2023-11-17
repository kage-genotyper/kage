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



def test_convert_to_multiallelic():
    matrix = SparseHaplotypeMatrix.from_nonsparse_matrix(
        [
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 0], # grouped
            [1, 0, 1], # grouped
            [0, 0, 1]
        ]
    )

    n_alleles_per_variant = np.array([2, 2, 3, 2])
    nonsparse = matrix.to_multiallelic(n_alleles_per_variant)

    correct = [
        [1, 1, 0],
        [0, 0, 0],
        [2, 1, 2],
        [0, 0, 1]
    ]

    assert np.all(nonsparse.to_matrix() == np.array(correct))


def test_sparse_haplotype_matrix_from_multiallelic_vcf():
    matrix = SparseHaplotypeMatrix.from_vcf("multiallelic.vcf")
    matrix = matrix.to_matrix()

    correct = np.array([
        [0, 0, 0, 1, 2, 0, 2, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 2]
    ])

    assert np.all(correct == matrix)


def test_sparse_haplotype_matrix_from_biallelic_vcf():
    matrix = SparseHaplotypeMatrix.from_vcf("biallelic.vcf")
    matrix = matrix.to_matrix()

    correct = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ])

    assert np.all(correct == matrix)


def test_to_biallelic():
    matrix = SparseHaplotypeMatrix.from_nonsparse_matrix(
        [
            [0, 1, 2, 1, 0],
            [0, 3, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1]
        ]
    )

    n_alleles_per_variant = np.array([3, 4, 2, 3])
    biallelic = matrix.to_biallelic(n_alleles_per_variant)

    correct = np.array([
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],

        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],

        [1, 0, 0, 0, 1],

        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0]
    ])

    assert np.all(biallelic.to_matrix() == correct)


def test_sparse_haplotype_matrix_with_missing_data():
    # missing data is encoded as 9
    missing_encoding = 127
    e = missing_encoding
    m = SparseHaplotypeMatrix.from_nonsparse_matrix(
        np.array([
            [0, e, 0, 2, 1, e],
            [0, 1, 0, 1, 0, e],
        ])
    )
    n_alleles_per_variant = np.array([3, 2])

    biallelic = m.to_biallelic(n_alleles_per_variant, missing_data_encoding=missing_encoding)

    correct = [
        [0, e, 0, 0, 1, e],
        [0, e, 0, 1, 0, e],
        [0, 1, 0, 1, 0, e]
    ]

    print(biallelic.to_matrix())
    assert np.all(biallelic.to_matrix() == correct)


#def test_sparse_haplotype_matrix_to_biallelic_with_missing_data():


def test_convert_biallelic_to_multiallelic():
    matrix = np.load("test_biallelic_haplotype_matrix.npy")
    n_alleles_per_variant = np.array([14])
    biallelic = SparseHaplotypeMatrix.from_nonsparse_matrix(matrix)
    multiallelic = biallelic.to_multiallelic(n_alleles_per_variant)
    print(multiallelic.to_matrix())
