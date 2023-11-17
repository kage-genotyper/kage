from kage.indexing.sparse_haplotype_matrix import SparseHaplotypeMatrix, GenotypeMatrix
from kage.models.helper_model import find_best_helper, create_combined_matrices, make_helper_model_from_genotype_matrix


def test_find_good_helper_variants_integration():
    """
    Test that good helper variants are found when variants are multiallelic
    We don't want to use helpers inside the same multiallelic variant
    """

    vcf_file_name = "tricky_variants21.vcf"
    haplotype_matrix = SparseHaplotypeMatrix.from_vcf(vcf_file_name)

    genotype_matrix = GenotypeMatrix.from_haplotype_matrix(haplotype_matrix)


    helper_model, combo_matrix = make_helper_model_from_genotype_matrix(
        genotype_matrix.matrix, None, dummy_count=1.0, window_size=100)

    print("Best helpers")
    print(helper_model)

    # no variants in the multiallelic variants should have best helper in the multiallelic
    #assert all([best_helper not in range(4, 73) for best_helper in helper_model[4:73]])