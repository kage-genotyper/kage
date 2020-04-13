from numba import jit
import numpy as np


@jit(nopython=True)
def letter_sequence_to_numeric(letter_sequence):
    numeric = np.zeros(len(letter_sequence), dtype=np.int64)

    i = 0
    for i in range(0, len(letter_sequence)):
        base = letter_sequence[i]
        if base == "A" or base == "a":
            numeric[i] = 0
        elif base == "C" or base == "c":
            numeric[i] = 1
        elif base == "T" or base == "t":
            numeric[i] = 2
        elif base == "G" or base == "g":
            numeric[i] = 3

    return numeric


if __name__ == "__main__":
    import timeit
    print(letter_sequence_to_numeric("AcaCTgactgactagactacggagactacgaCATACGGTggACTACG"))
    time = timeit.timeit('letter_sequence_to_numeric("CAGTGGGAAGATACAGGAGGCTATCTGACCCACACTGGACTGGGTGTAAGGGAAATGAGACCCCTGTGTGGCAGACACTGAAGACTGTTTTCAAAATGCCATTTACATCCCACCCATTTACATCCCacccatttacatctacccacagtg")', "from __main__ import letter_sequence_to_numeric", number=1090)
    print(time)
