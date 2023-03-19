import numpy as np
import logging


class NodeCounts:
    properties = {"node_counts"}

    def __init__(self, node_counts=None):
        self.node_counts = node_counts

    def to_file(self, file_name):
        np.save(file_name, self.node_counts)

    def get_node_count_array(self, min_nodes=None):
        if min_nodes is not None and min_nodes >= len(self.node_counts):
            # pad with zero counts
            logging.info("Padding node counts to %d" % min_nodes)
            return np.pad(self.node_counts, (0, min_nodes-len(self.node_counts)+1))

        return self.node_counts

    def __getitem__(self, item):
        return self.node_counts[item]

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(data)
