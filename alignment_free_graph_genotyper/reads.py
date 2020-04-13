
class Reads:
    def __init__(self, reads):
        self.reads = reads

    def __next__(self):
        return self.__next_()

    def __iter__(self):
        return self.reads.__iter__()

    @classmethod
    def from_fasta(cls, fasta_file_name):
        reads = (line.strip() for line in open(fasta_file_name) if not line.startswith(">"))
        return cls(reads)

