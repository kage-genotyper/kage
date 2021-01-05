import numpy as np

class NodeCountModel:
    def __init__(self, node_counts_following_node, node_counts_not_following_node, average_coverage=1):
        self.node_counts_following_node = node_counts_following_node
        self.node_counts_not_following_node = node_counts_not_following_node

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(data["node_counts_following_node"], data["node_counts_not_following_node"])

    def to_file(self, file_name):
        np.savez(file_name, node_counts_following_node=self.node_counts_following_node,
            node_counts_not_following_node=self.node_counts_not_following_node)
