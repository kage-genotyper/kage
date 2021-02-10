import logging

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

def read_chunks_from_fasta(fasta_file_name, chunk_size=10):
    logging.info("Read chunks")
    file = open(fasta_file_name)
    out = []
    i = 0
    for line in file:
        if line.startswith(">"):
            continue

        if i % 500000 == 0:
            logging.info("Read %d lines" % i)

        out.append(line.strip())
        i += 1
        if i >= chunk_size and chunk_size > 0:
            yield out
            out = []
            i = 0
    yield out

