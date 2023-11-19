from kage.genotyping.multiallelic import postprocess_multiallelic_calls
import numpy as np
from npstructures import RaggedArray
from kage.indexing.main import MultiAllelicMap


def test_posprocess_multiallelic_calls():
    genotypes = np.array([1, 2, 2, 2])  # 0/0, 1/1, 1/1, 1/1
    probs = np.log(np.array([
        [0.98, 0.01, 0.01],
        [0.3, 0.3, 0.4],
        [0.05, 0.05, 0.9],
        [0.01, 0.01, 0.98],
    ]))
    multiallic_map = MultiAllelicMap.from_n_alleles_per_variant(
        np.array([2, 4])
    )
    print(multiallic_map)
    print(multiallic_map.ravel())

    new_genotypes, new_probs = postprocess_multiallelic_calls(genotypes, multiallic_map, probs)

    print(new_genotypes)
    assert np.all(new_genotypes == np.array([1, 1, 1, 2]))


def test_posprocess_multiallelic_calls2():
    # case where genotype with highest prob is 0/0 but there is another 0/1 with high prob
    # this other should not become 0/0
    genotypes = np.array([1, 2, 2, 2])  # 0/0, 1/1, 1/1, 1/1
    probs = np.log(np.array([
        [0.99, 0.01, 0.01],
        [0.3, 0.4, 0.3],
        [0.01, 0.98, 0.01],  # this should be kept
        [0.3, 0.4, 0.3],
    ]))
    multiallic_map = MultiAllelicMap.from_n_alleles_per_variant(
        np.array([5])
    )
    print(multiallic_map)
    print(multiallic_map.ravel())

    new_genotypes, new_probs = postprocess_multiallelic_calls(genotypes, multiallic_map, probs)

    print(new_genotypes)
    assert np.all(new_genotypes == np.array([1, 1, 2, 1]))


def test_posprocess_multiallelic_calls3():
    # case where genotype with highest prob is 0/0, and this variant also has highest non-0/0-prob. Then all should be 0/0
    genotypes = np.array([1, 2, 2, 2])  # 0/0, 1/1, 1/1, 1/1
    probs = np.log(np.array([
        [0.5, 0.49, 0.01],
        [0.3, 0.4, 0.3],
        [0.3, 0.4, 0.3],  # this should be kept
        [0.3, 0.4, 0.3],
    ]))
    multiallic_map = MultiAllelicMap.from_n_alleles_per_variant(
        np.array([5])
    )
    new_genotypes, new_probs = postprocess_multiallelic_calls(genotypes, multiallic_map, probs)
    assert np.all(new_genotypes == np.array([1, 1, 1, 1]))

