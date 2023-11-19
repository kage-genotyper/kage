import numpy as np
import numba


def get_filter_of_variants_part_of_multiallelic_variant(n_alleles_per_multiallelic_variant):
    """
    Returns an np.array of length of number of biallelic variants
    True if the biallelic variant is part of a multiallelic variant
    """
    filter = np.zeros(np.sum(n_alleles_per_multiallelic_variant-1), dtype=bool)

    @numba.jit(nopython=True)
    def fill(f, n_alleles):
        biallelic_id = 0
        for multiallelic in n_alleles:
            for allele in range(multiallelic-1):
                if multiallelic > 2:
                    f[biallelic_id] = True
                biallelic_id += 1
        return f

    fill(filter, n_alleles_per_multiallelic_variant)
    return filter
