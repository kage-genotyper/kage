from graph_kmer_index.cython_kmer_index import CythonKmerIndex
import logging
import numpy as np
cimport numpy as np
cimport cython
import time


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list chain(np.ndarray[np.uint64_t] ref_offsets,  np.ndarray[np.uint64_t] read_offsets,  np.ndarray[np.uint64_t] nodes, unsigned long[:] kmers):

    cdef np.ndarray[np.uint64_t] potential_chain_start_positions = ref_offsets - read_offsets
    cdef int i, start, end


    cdef long[:] sorting = np.argsort(potential_chain_start_positions)
    #ref_offsets = ref_offsets[sorting]
    read_offsets = read_offsets[sorting]
    nodes = nodes[sorting]
    potential_chain_start_positions = potential_chain_start_positions[sorting]


    cdef list chains = []
    cdef float score
    cdef int current_start = 0
    cdef int prev_position = potential_chain_start_positions[0]
    for i in range(1, potential_chain_start_positions.shape[0]):
        if potential_chain_start_positions[i] >= prev_position + 2:
            score = 0.0  #np.unique(read_offsets[current_start:i]).shape[0]
            chains.append([potential_chain_start_positions[current_start], nodes[current_start:i], score, kmers])
            current_start = i
        prev_position = potential_chain_start_positions[i]
    score = 0.0  #np.unique(read_offsets[current_start:]).shape[0]
    chains.append([potential_chain_start_positions[current_start], nodes[current_start:], score, kmers])

    return chains


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int64_t, ndim=1] letter_sequence_to_numeric(np.ndarray[np.int8_t, ndim=1] letter_sequence):
#cdef letter_sequence_to_numeric(np.ndarray[np.int8_t, ndim=1] letter_sequence):
    cdef np.ndarray[np.uint64_t] numeric = np.zeros(letter_sequence.shape[0], dtype=np.uint64)
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
cdef int set_intersection(int[:] set_array, unsigned long[:] other_array):
    cdef int n = 0
    cdef int i = 0
    for i in range(other_array.shape[0]):
        if set_array[other_array[i] % 100000007] == 1:
            n += 1
            set_array[other_array[i] % 100000007] = 0
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int set_intersection_noslice(int[:] set_array, unsigned long[:] other_array, unsigned long start, unsigned long stop):
    cdef int n = 0
    cdef unsigned long i = 0
    for i in range(start, stop):
        if set_array[other_array[i] % 100000007] == 1:
            n += 1
            set_array[other_array[i] % 100000007] = 0
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned long[:] get_kmers(np.ndarray[np.uint64_t] numeric_read, np.ndarray[np.uint64_t] power_array):
#def get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
    return np.convolve(numeric_read, power_array, mode='valid')

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned long[:] get_spaced_kmers(np.ndarray[np.uint64_t] numeric_read, np.ndarray[np.uint64_t] power_array, int spacing):
#def get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
    return np.convolve(numeric_read, power_array, mode='valid')[::spacing]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef map(reads, kmer_index, ref_kmers, int k, int k_short, int max_node_id, unsigned long[:] ref_index_index, unsigned long[:] ref_index_kmers):
    start_time = time.time()
    cdef int n_reads = len(reads)
    cdef np.ndarray[np.uint64_t, ndim=1] numeric_read, reverse_read
    cdef unsigned long[:] kmers, short_kmers, kmer_hashes
    cdef np.ndarray[np.uint64_t] power_array = np.power(4, np.arange(0, k, dtype=np.uint64))
    cdef np.ndarray[np.uint64_t] power_array_short = np.power(4, np.arange(0, k_short, dtype=np.uint64))
    cdef np.ndarray[np.uint64_t] node_hits
    cdef np.ndarray[np.uint64_t] ref_offsets_hits
    cdef np.ndarray[np.uint64_t] read_offsets_hits
    cdef np.ndarray[np.uint64_t, ndim=2] index_lookup_result
    cdef list forward_and_reverse_chains
    cdef float chain_score
    cdef unsigned long ref_start, ref_end
    cdef unsigned long approx_read_length = 150
    cdef np.ndarray[np.int64_t] best_chain_kmers
    cdef int l, n_hits
    #cdef np.ndarray[np.int64_t] reference_kmers = ref_kmers.reference_kmers
    cdef float best_score = 0
    cdef np.ndarray[np.uint16_t] node_counts = np.zeros(max_node_id, dtype=np.uint16)

    cdef unsigned long[:] mapping_positions = np.zeros(n_reads, dtype=np.uint64)
    cdef int n_chains
    cdef int n_unique_short_kmers
    cdef set short_kmers_set
    cdef int u
    cdef int chain_score_int

    cdef int[:] short_kmers_array_set = np.zeros(100000007, dtype=np.int32)

    logging.info("Mapping %d reads" % n_reads)

    prev_time = time.time()
    cdef int read_number = -1
    for read in reads:
        if read_number % 1000 == 0:
            logging.info("%d reads processed in %.5f sec" % (read_number, time.time() - prev_time))
            prev_time = time.time()
        read_number += 1

        numeric_read = letter_sequence_to_numeric(np.array(list(read), dtype="|S1").view(np.int8))
        reverse_read = numeric_read[::-1]
        #forward_and_reverse_chains = []

        for l in range(2):
            if l == 0:
               kmers = get_kmers(numeric_read, power_array)
               short_kmers = get_kmers(numeric_read, power_array_short)
               #short_kmers = get_spaced_kmers(numeric_read, power_array_short, k_short)
            else:
               kmers = get_kmers(reverse_read, power_array)
               short_kmers = get_kmers(reverse_read, power_array_short)
               #short_kmers = get_spaced_kmers(reverse_read, power_array_short, k_short)


            #short_kmers_set = set(short_kmers)
            #n_unique_short_kmers = len(short_kmers_set)

            index_lookup_result = kmer_index.get(kmers)
            n_hits = index_lookup_result.shape[1]
            if n_hits == 0:
                continue

            node_hits = index_lookup_result[0,:]
            ref_offsets_hits = index_lookup_result[1,:]
            read_offsets_hits = index_lookup_result[2,:]

            chains = chain(ref_offsets_hits, read_offsets_hits, node_hits, kmers)

            n_chains = len(chains)

            for c in range(n_chains):
                for u in range(short_kmers.shape[0]):
                    short_kmers_array_set[short_kmers[u] % 100000007] = 1

                ref_start = chains[c][0]
                #ref_end = ref_start + approx_read_length
                #chains[c][2] = len(short_kmers_set.intersection(reference_kmers[ref_start:ref_end-k_short]))
                #chains[c][2] = len(short_kmers_set.intersection(ref_kmers.get_between(ref_start, ref_end-k_short)))
                #logging.info("Start, end: %d/%d" % (ref_index_index[ref_start], ref_index_index[ref_end-k_short]))
                #assert ref_index_index[ref_end] - ref_index_index[ref_start] < 500, "Very long ref index segment for ref start/end %d / %d" % (ref_start, ref_end)
                #if chains[c][2] <= 0 and n_chains > 10:
                #    chains[c][2] = 0.0
                #    continue
                #chains[c][2] = len(short_kmers_set.intersection(ref_index_kmers[ref_index_index[ref_start]:ref_index_index[ref_end-k_short]]))
                #chains[c][2] = set_intersection(short_kmers_array_set, ref_index_kmers[ref_index_index[ref_start]:ref_index_index[ref_end-k_short]])    # n_unique_short_kmers
                chain_score_int = set_intersection_noslice(short_kmers_array_set, ref_index_kmers, ref_index_index[ref_start], ref_index_index[ref_start + approx_read_length - k_short])
                #chains[c][2] = chain_score_int

                if chain_score_int >= best_score:
                    best_score = chain_score_int
                    best_chain = chains[c]


            #logging.info("N chains: %d" % len(chains))
            for u in range(short_kmers.shape[0]):
                short_kmers_array_set[short_kmers[u] % 100000007] = 0

            #forward_and_reverse_chains.extend(chains)


        # Find best chain
        """
        best_chain_kmers = None
        best_chain = None
        for c in range(len(forward_and_reverse_chains)):
            if forward_and_reverse_chains[c][2] >= best_score:
                best_score = forward_and_reverse_chains[c][2]
                best_chain = forward_and_reverse_chains[c]
        """

        #if time.time() - prev_time > 1:
        #    logging.info("LONG TIME! Read: %s" % read)
        #logging.info("Best score: %.3f" % best_score)

        if best_score < 0.4: # * 150 / 15:  #  * (150 - k_short):
            continue

        best_score = 0.0
        if best_chain is None:
            continue


        # Increase node counts
        for l in range(best_chain[1].shape[0]):
            node_counts[best_chain[1][l]] += 1

        mapping_positions[read_number] = best_chain[0]

    logging.info("Time spent: %.4f" % (time.time() - start_time))
    return mapping_positions, node_counts




