import logging
from shared_memory_wrapper import (
    SingleSharedArray,
    to_shared_memory,
    from_shared_memory,
)
import random
import numpy as np
import gzip
import time


class Reads:
    def __init__(self, reads):
        self.reads = reads

    def __next__(self):
        return self.__next_()

    def __iter__(self):
        return self.reads.__iter__()

    @classmethod
    def from_fasta(cls, fasta_file_name):
        reads = (
            line.strip() for line in open(fasta_file_name) if not line.startswith(">")
        )
        return cls(reads)


def read_chunks_from_fasta(
    fasta_file_name,
    chunk_size=10,
    include_read_names=False,
    assign_numeric_read_names=False,
    write_to_shared_memory=False,
    max_read_length=150,
    save_as_bytes=False,
):

    is_gzipped = False
    if fasta_file_name.endswith(".gz"):
        file = gzip.open(fasta_file_name)
        is_gzipped = True
    else:
        file = open(fasta_file_name)

    out = []
    i = 0
    read_id = 0
    current_read_name = None

    if save_as_bytes:
        out_array = np.empty(
            chunk_size, dtype="|S" + str(max_read_length)
        )  # max_read_length bytes for each element
    else:
        out_array = np.empty(chunk_size, dtype="<U" + str(max_read_length))

    prev_time = time.time()

    for line in file:
        if is_gzipped:
            line = line.decode("utf-8")

        if line.startswith(">"):
            if include_read_names:
                if assign_numeric_read_names:
                    current_read_name = read_id
                else:
                    current_read_name = line[1:].strip()
            continue

        if i % 500000 == 0:
            logging.info("Read %d lines" % i)

        if len(line) < 31:
            logging.warning("Fasta sequence %s is short, skipping." % line.strip())
            continue

        if include_read_names:
            assert False, "Not supported, not rewritten to using np arrays"
            out.append((current_read_name, line.strip()))
        else:
            # out.append(line.strip())
            out_array[i] = line.strip()

        i += 1
        read_id += 1
        if i >= chunk_size and chunk_size > 0:
            logging.info(
                "Returning chunk of %d reads (took %.5f sec)"
                % (chunk_size, time.time() - prev_time)
            )
            prev_time = time.time()
            if write_to_shared_memory:
                out = SingleSharedArray(out_array)
                shared_name = "shared_array_" + str(
                    random.randint(1000000, 10000000000)
                )
                to_shared_memory(out, shared_name)
                yield shared_name
            else:
                yield out_array
            out = []
            out_array = np.empty(chunk_size, dtype="<U" + str(max_read_length))
            i = 0

    logging.info(
        "Returning chunk of %d reads (took %.5f sec)"
        % (chunk_size, time.time() - prev_time)
    )

    if write_to_shared_memory:
        out = SingleSharedArray(out_array[0:i])
        shared_name = "shared_array_" + str(random.randint(1000000, 10000000000))
        to_shared_memory(out, shared_name)
        yield shared_name
    else:
        yield out_array[0:i]


def read_chunks_from_fastq(
    fastq_file_name,
    chunk_size=10,
    include_read_names=False,
    assign_numeric_read_names=False,
):
    file = open(fastq_file_name)
    out = []
    i = 0
    current_read_name = None
    for line_number, line in enumerate(file):
        if line.startswith("@"):
            assert line_number % 4 == 0
            if include_read_names:
                if assign_numeric_read_names:
                    current_read_name = i
                else:
                    current_read_name = line[1:].strip()
            continue

        # Skip quality and + line
        if line_number % 4 >= 2:
            continue

        if i % 500000 == 0:
            logging.info("Read %d lines" % i)

        assert line_number % 4 == 1

        if include_read_names:
            out.append((current_read_name, line.strip()))
        else:
            out.append(line.strip())

        i += 1
        if i >= chunk_size and chunk_size > 0:
            logging.info("Yielding %d reads" % (len(out)))
            yield out
            out = []
            i = 0
    yield out
