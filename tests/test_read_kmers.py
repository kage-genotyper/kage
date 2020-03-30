from alignment_free_graph_genotyper.genotyper import ReadKmers
from graph_kmer_index import letter_sequence_to_numeric, kmer_to_hash_fast
import numpy as np


def test_from_fasta():

    reads = ["CTGAAAC",
             "ATTAGAC"]

    with open("tmp.fasta", "w") as f:
        for read in reads:
            f.write(">read\n" + read + "\n")

    print(kmer_to_hash_fast(letter_sequence_to_numeric("CTG"), 3))
    readkmers = list(ReadKmers.from_fasta_file("tmp.fasta", k=3))


    print(readkmers)

    assert list(readkmers[0])[0] == [16+8+3]
    assert list(readkmers[0])[1] == [kmer_to_hash_fast(letter_sequence_to_numeric("TGA"), 3)]
    assert list(readkmers[0]) == [27, 44, 48, 0, 1]
    for r in readkmers:
        print(r)

    print("Test: " , kmer_to_hash_fast(np.array([2, 2, 3, 3, 2, 0, 0, 2, 1, 2, 0, 1, 0, 2, 3, 0, 0, 1, 0, 2, 2, 2, 3, 3, 2, 0, 0, 2, 1, 2, 0]), 31))
    # Longer reads
    reads = \
        ["ACAtgAACAtttggtAATCTACAtgAACAtttggtAATCTACAtgAACAtttggtAATCTACAtgAACAtttggtAATCTACAtgAACAtttggtAATCT",
         "ACACTGGTACGGACTGGACTACAtgAACAtttggtAATCTACAtgAACAtttggtAATCTACAtgAACAtttggtAATCTACAtgAACAtttggtAATCT"]
    with open("tmp.fasta", "w") as f:
        for read in reads:
            f.write(">read\n" + read + "\n")
    k = 31
    readkmers = list(ReadKmers.from_fasta_file("tmp.fasta", k=31))
    print(readkmers)

    assert readkmers[0][20] == kmer_to_hash_fast(letter_sequence_to_numeric(reads[0][20:20+k]), k)

if __name__ == "__main__":
    test_from_fasta()


