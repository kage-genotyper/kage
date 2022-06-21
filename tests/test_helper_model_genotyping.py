import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from kage.combination_model_genotyper import CombinationModelGenotyper
from obgraph.variant_to_nodes import VariantToNodes
from kage.node_counts import  NodeCounts
from kage.node_count_model import NodeCountModelAdvanced
from obgraph.variants import VcfVariants
from kage.sampling_combo_model import LimitedFrequencySamplingComboModel

def test():
    variant_to_nodes = VariantToNodes(np.array([0, 2]), np.array([1, 3]))
    node_counts = NodeCounts(np.array([4, 3, 10, 0]))



    combo_matrix = np.array([
        np.zeros((3, 3)),
        np.array([[0.21643952, 0.00000853, 0.00000594],
         [0.00257945, 0.4476593,  0.00050901],
        [0.00022379,  0.00488694,  0.32768752,]])
    ]
    )
    combo_matrix = np.array([
        np.zeros((3, 3)),
        [[0.20867266, 0.00922959, 0.00097657],
         [0.00016128, 0.43460627, 0.01792008],
        [0.00022378, 0.00053976, 0.32767001]]
        ])


    helpers = np.array([1, 0])
    #node_count_model = NodeCountModelAdvanced.from_dict_of_frequencies({}, 4)
    model_ref = LimitedFrequencySamplingComboModel.create_naive(2)
    model_var = LimitedFrequencySamplingComboModel.create_naive(2)

    genotyper = CombinationModelGenotyper(
        [model_ref, model_var], 0, 3, variant_to_nodes, node_counts, helper_model=helpers, helper_model_combo=combo_matrix
    )

    genotypes, probs, count_probs = genotyper.genotype()
    genotypes = VcfVariants.translate_numeric_genotypes_to_literal(genotypes)


test()
