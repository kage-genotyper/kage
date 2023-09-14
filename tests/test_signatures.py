import pytest
from kage.indexing.signatures import MultiAllelicSignatures


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


