import numpy as np


class TrickyVariants:
    properties = {"tricky_variants"}

    def __init__(self, tricky_variants):
        self.tricky_variants = tricky_variants

    @classmethod
    def from_file(cls, file_name):
        return cls(np.load(file_name))

    def to_file(self, file_name):
        np.save(file_name, self.tricky_variants)
