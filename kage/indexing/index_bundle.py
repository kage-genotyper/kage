import logging
import numpy as np
from obgraph.variant_to_nodes import VariantToNodes
from obgraph.numpy_variants import NumpyVariants
from graph_kmer_index import KmerIndex
from shared_memory_wrapper import from_file, to_file
from ..indexing.tricky_variants import TrickyVariants
from ..models.helper_model import HelperVariants, CombinationMatrix


class IndexBundle:
    def __init__(self, index):
        if isinstance(index, list):
            index = dict(index)
        self._index = index

    def __getitem__(self, e):
        return self._index[e]

    def __setitem__(self, e, v):
        self._index[e] = v

    @classmethod
    def from_args(cls, args):
        return cls([
            ("VariantToNodes", VariantToNodes.from_file(args.variant_to_nodes)),
            ("NumpyVariants", NumpyVariants.from_file(args.numpy_variants)),
            ("CountModel", from_file(args.count_model)),
            ("TrickyVariants", TrickyVariants.from_file(args.tricky_variants)),
            ("HelperVariants", HelperVariants.from_file(args.helper_model)),
            ("CombinationMatrix", CombinationMatrix.from_file(args.helper_model_combo_matrix)),
            ("KmerIndex", KmerIndex.from_file(args.kmer_index))
        ])

    @classmethod
    def from_file(cls, file_name, skip=None):
        if skip is not None:
            logging.warning("SKip option does not work")
        return from_file(file_name)

    def to_file(self, file_name, compress=True):
        return to_file(self, file_name, compress=compress)

    @property
    def indexes(self):
        # for backwards compatibility
        return self

