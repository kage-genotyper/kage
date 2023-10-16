import time

import numpy as np
import pytest
from kage.indexing.signatures import MultiAllelicSignatures, MultiAllelicSignatureFinderV2
import awkward as ak

def test_multiallelic_signatures_from_multiple():

    s1_list = [
        [
            [0, 1],
            [2, 3]
        ],
        [
            [1],
            [10],
            [11]
        ]
    ]
    s1 = MultiAllelicSignatures.from_list(s1_list)

    s2_list = [
        [
            [5],
            [10]
        ]
    ]
    s2 = MultiAllelicSignatures.from_list(s2_list)

    merged = MultiAllelicSignatures.from_multiple([s1, s2])

    print(merged.to_list())
    assert merged.to_list() == s1_list + s2_list




def test_manually_find_kmers():
    kmers = ak.Array(np.array([
        # allele
        [
            # path
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            # path
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ],
        [
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        ]
    ], dtype=np.uint64))

    scores = ak.Array([
        allele[0] for allele in kmers
    ])

    scores = ak.zeros_like(scores, dtype=float)
    print(kmers)
    print(scores)

    result = MultiAllelicSignatureFinderV2.manually_find_kmers(kmers, scores)

    # first kmers is chosen, 12 is first unique for allele 2
    assert result.signatures.tolist() == [[[0], [12]]]


# Only for profiling
def test_manually_find_kmers_large():
    window_size = 3600
    n_paths = 128
    n_alleles = 86

    kmers = ak.Array(np.array([
        # allele
        [
            # path
            np.random.randint(0, 1000, size=window_size, dtype=np.uint64)
            for _ in range(n_paths)
        ]
        for _ in range(n_alleles)

    ], dtype=np.uint64))

    scores = ak.Array([
        allele[0] for allele in kmers
    ])

    scores = ak.zeros_like(scores, dtype=float)

    t0 = time.perf_counter()
    result = MultiAllelicSignatureFinderV2.manually_find_kmers(kmers, scores)
    print(time.perf_counter()-t0)

    #assert False




def test_ak_sum():
    # variants x alleles x paths x windows
    paths_x_windows = []
    pass

