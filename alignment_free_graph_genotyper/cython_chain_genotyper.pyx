import logging
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list chain(np.ndarray[np.int64_t] ref_offsets, np.ndarray[np.int64_t] read_offsets, np.ndarray[np.int64_t] nodes):
    potential_chain_start_positions = ref_offsets - read_offsets
    sorting = np.argsort(potential_chain_start_positions)
    ref_offsets = ref_offsets[sorting]
    nodes = nodes[sorting]
    potential_chain_start_positions = potential_chain_start_positions[sorting]


    cdef chains = []
    cdef int i, start, end
    cdef int current_start = 0
    cdef int prev_position = potential_chain_start_positions[0]
    for i in range(1, len(potential_chain_start_positions)):
        if potential_chain_start_positions[i] >= prev_position + 2:
            chains.append([potential_chain_start_positions[current_start], nodes[current_start:i], 0])
            current_start = i
        prev_position = potential_chain_start_positions[i]
    chains.append([potential_chain_start_positions[current_start], nodes[current_start:], 0])

    return chains

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int64_t, ndim=1] letter_sequence_to_numeric(np.ndarray[np.int8_t, ndim=1] letter_sequence):
#cdef letter_sequence_to_numeric(np.ndarray[np.int8_t, ndim=1] letter_sequence):
    cdef np.ndarray[np.int64_t] numeric = np.zeros(letter_sequence.shape[0], dtype=np.int)
    cdef int i = 0
    cdef int base
    for i in range(0, letter_sequence.shape[0]):
        base = letter_sequence[i]
        if base == 97 or base == 65:
            numeric[i] = 0
        elif base == 99 or base == 67:
            numeric[i] = 1
        elif base == 116 or base == 84:
            numeric[i] = 2
        elif base == 103 or base == 71:
            numeric[i] = 3
    return numeric

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int64_t] get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
#def get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
    return np.convolve(numeric_read, power_array, mode='valid')


@cython.boundscheck(False)
@cython.wraparound(False)
def run(reads_file_name,
          np.ndarray[np.int64_t] hashes,
          np.ndarray[np.int64_t] hashes_to_index,
          np.ndarray[np.int64_t] n_kmers,
          np.ndarray[np.uint32_t] nodes,
          np.ndarray[np.uint32_t] ref_offsets,
          np.ndarray[np.int64_t] reference_kmers,
          ):

    cdef int k = 31
    cdef int k_short = 16
    cdef np.ndarray[np.int64_t] power_array = np.power(4, np.arange(0, k))
    cdef np.ndarray[np.int64_t] power_array_short = np.power(4, np.arange(0, k_short))

    file = open(reads_file_name)
    cdef list chains = []
    cdef np.ndarray[np.int64_t, ndim=1] numeric_read, reverse_read
    cdef np.ndarray[np.int64_t, ndim=1] kmers, short_kmers, kmer_hashes
    cdef int index_position, i
    cdef int n_total_hits = 0
    cdef int hash
    cdef int n

    cdef np.ndarray[np.int64_t] found_nodes
    cdef np.ndarray[np.int64_t] found_ref_offsets
    cdef np.ndarray[np.int64_t] found_read_offsets

    cdef int l, c
    cdef int counter = 0
    cdef int n_local_hits, j
    cdef list forward_and_reverse_chains
    cdef float chain_score
    cdef int ref_start, ref_end
    cdef int approx_read_length = 150
    cdef np.ndarray[np.int64_t] local_reference_kmers
    cdef set short_kmers_set
    cdef float best_score = 0
    cdef list best_chain
    cdef read_number = -1
    cdef np.ndarray[np.int64_t] chain_positions = np.zeros(100000, dtype=np.int64)

    for line in file:
        if line.startswith(">"):
            continue
        read_number += 1

        numeric_read = letter_sequence_to_numeric(np.array(list(line.strip()), dtype="|S1").view(np.int8))
        reverse_read = numeric_read[::-1]
        forward_and_reverse_chains = []

        for l in range(2):
            if l == 0:
               kmers = get_kmers(numeric_read, power_array)
               short_kmers = get_kmers(numeric_read, power_array_short)
            else:
               kmers = get_kmers(reverse_read, power_array)
               short_kmers = get_kmers(reverse_read, power_array_short)

            short_kmers_set = set(short_kmers)

            kmer_hashes = np.mod(kmers, 452930477)
            n = kmers.shape[0]
            n_total_hits = 0

            # First find number of hits
            for i in range(n):
                hash = hashes[kmer_hashes[i]]
                if hash == 0:
                    continue
                n_total_hits += n_kmers[hash]

            found_nodes = np.zeros(n_total_hits, dtype=np.int)
            found_ref_offsets = np.zeros(n_total_hits, dtype=np.int)
            found_read_offsets = np.zeros(n_total_hits, dtype=np.int)

            # Get the actual hits
            counter = 0

            for i in range(n):
                hash = hashes[kmer_hashes[i]]
                if hash == 0:
                    continue

                index_position = hashes_to_index[hash]
                n_local_hits = n_kmers[hash]

                if n_local_hits == 0:
                    continue

                for j in range(n_local_hits):
                    found_nodes[counter] = nodes[index_position+j]
                    found_ref_offsets[counter] = ref_offsets[index_position+j]
                    found_read_offsets[counter] = i
                    counter += 1

            if len(found_nodes) == 0:
                continue

            #print(found_nodes)
            chains = chain(found_ref_offsets, found_read_offsets, found_nodes)

            # score chains
            for c in range(len(chains)):
                ref_start = chains[c][0]
                ref_end = ref_start + approx_read_length
                local_reference_kmers = reference_kmers[ref_start:ref_end-k_short]
                chain_score = len(short_kmers_set.intersection(local_reference_kmers)) / len(short_kmers_set)
                chains[c][2] = chain_score

            forward_and_reverse_chains.extend(chains)
            #print(chains)

        # Find best chain
        best_score = 0
        for c in range(len(forward_and_reverse_chains)):
            if forward_and_reverse_chains[c][2] >= best_score:
                best_score = forward_and_reverse_chains[c][2]
                best_chain = forward_and_reverse_chains[c]

        chain_positions[read_number] = best_chain[0]

        #print("%d\t%d\t%s" % (read_number, best_chain[0], best_chain))

    return chain_positions






