import logging
logging.basicConfig(level=logging.ERROR)
import pytest
from kage.node_count_model import NodeCountModelAdvanced
from kage.combomodel import ComboModel
from kage.models import ComboModelBothAlleles, HelperModel
import numpy as np
np.seterr(all="ignore")


@pytest.mark.skip("Outdated model")
def test_simple():
    coverage = 0.85 * 0.75 * 15 / 2

    nodes = np.array([0, 1, 2, 3])
    dummy_frequencies = {node: [0.02] for node in nodes}
    #dummy_frequencies = {}

    count_model = NodeCountModelAdvanced.from_dict_of_frequencies(dummy_frequencies, 4)

    ref_nodes = [0, 2]
    alt_nodes = [1, 3]


    models = [ComboModel.from_counts(coverage, count_model.frequencies[nodes],
                                     count_model.frequencies_squared[nodes],
                                     count_model.has_too_many[nodes],
                                     count_model.certain[nodes],
                                     count_model.frequency_matrix[nodes])
              for nodes in (ref_nodes, alt_nodes)]

    combination_model_both = ComboModelBothAlleles(*models)


    node_counts_ref = np.array([5, 3])
    node_counts_alt = np.array([0, 3])
    probs_homo_ref = combination_model_both.logpmf(node_counts_ref, node_counts_alt, 0)
    probs_hetero = combination_model_both.logpmf(node_counts_ref, node_counts_alt, 1)
    probs_homo_alt = combination_model_both.logpmf(node_counts_ref, node_counts_alt, 2)


    print(probs_homo_ref)
    print(probs_hetero)
    print(probs_homo_alt)


