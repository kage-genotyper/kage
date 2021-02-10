import numpy as np

class NodeCounts:
    def __init__(self, node_counts):
        self.node_counts = node_counts

    def to_file(self, file_name):
        np.save(file_name, self.node_counts)

    def get_node_count_array(self):
        return self.node_counts

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npy")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(data)
