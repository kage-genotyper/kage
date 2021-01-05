from graph_kmer_index.cython_kmer_index import CythonKmerIndex
import logging
import numpy as np
cimport numpy as np
cimport cython
import time


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list chain(np.ndarray[np.uint64_t] ref_offsets,  np.ndarray[np.uint64_t] read_offsets,  np.ndarray[np.uint64_t] nodes, unsigned long[:] kmers, np.ndarray[np.uint64_t] frequencies, np.ndarray[np.uint64_t] allele_frequencies):

    cdef np.ndarray[np.uint64_t] potential_chain_start_positions = ref_offsets - read_offsets
    cdef int i, start, end


    cdef long[:] sorting = np.argsort(potential_chain_start_positions)
    #ref_offsets = ref_offsets[sorting]
    read_offsets = read_offsets[sorting]
    nodes = nodes[sorting]
    potential_chain_start_positions = potential_chain_start_positions[sorting]
    frequencies = frequencies[sorting]
    allele_frequencies = allele_frequencies[sorting]


    cdef list chains = []
    cdef float score
    cdef unsigned long current_start = 0
    cdef unsigned long prev_position = potential_chain_start_positions[0]
    for i in range(1, potential_chain_start_positions.shape[0]):
        if potential_chain_start_positions[i] >= prev_position + 25:
            score = np.sum(1 / frequencies[np.unique(read_offsets[current_start:i], True)[1]])  #.shape[0]
            #score = np.sum(allele_frequencies[np.unique(read_offsets[current_start:i], True)[1]])  #.shape[0]
            #score = np.sum(allele_frequencies[np.unique(read_offsets[current_start:i], True)[1]] / frequencies[np.unique(read_offsets[current_start:i], True)[1]])  #.shape[0]
            #print(allele_frequencies[np.unique(read_offsets[current_start:i], True)[1]])  #.shape[0]
            chains.append([potential_chain_start_positions[current_start], nodes[current_start:i], score, kmers])
            current_start = i
        prev_position = potential_chain_start_positions[i]
    #score = np.unique(read_offsets[current_start:]).shape[0]
    score = np.sum(1 / frequencies[np.unique(read_offsets[current_start:], True)[1]])
    #score = np.sum(allele_frequencies[np.unique(read_offsets[current_start:], True)[1]])
    #score = np.sum(allele_frequencies[np.unique(read_offsets[current_start:], True)[1]] / frequencies[np.unique(read_offsets[current_start:], True)[1]])
    chains.append([potential_chain_start_positions[current_start], nodes[current_start:], score, kmers])

    return chains


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.uint64_t] letter_sequence_to_numeric(np.ndarray[np.int8_t, ndim=1] letter_sequence):
#cdef letter_sequence_to_numeric(np.ndarray[np.int8_t, ndim=1] letter_sequence):
    cdef np.ndarray[np.uint64_t] numeric = np.zeros(letter_sequence.shape[0], dtype=np.uint64)
    cdef int i = 0
    cdef int base
    for i in range(0, letter_sequence.shape[0]):
        base = letter_sequence[i]
        if base == 97 or base == 65:
            numeric[i] = <np.uint64_t> 0
        elif base == 99 or base == 67:
            numeric[i] = <np.uint64_t> 1
        elif base == 116 or base == 84:
            numeric[i] = <np.uint64_t> 2
        elif base == 103 or base == 71:
            numeric[i] = <np.uint64_t> 3
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
cdef int set_intersection_noslice(int[:] set_array, unsigned int[:] other_array, unsigned int start, unsigned int stop):
    cdef int n = 0
    cdef unsigned long i = 0
    for i in range(start, stop):
        if set_array[other_array[i] % 100000007] == 1:
            n += 1
            set_array[other_array[i] % 100000007] = 0
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.uint64_t, ndim=1] get_kmers(np.ndarray[np.uint64_t] numeric_read, np.ndarray[np.uint64_t] power_array):
#def get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
    return np.convolve(numeric_read, power_array, mode='valid')

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned long[:] get_spaced_kmers(np.ndarray[np.uint64_t] numeric_read, np.ndarray[np.uint64_t] power_array, int spacing):
#def get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
    return np.convolve(numeric_read, power_array, mode='valid')[::spacing]

#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef map(reads, kmer_index, ref_kmers, int k, int k_short, int max_node_id, unsigned int[:] ref_index_kmers, reverse_index):
    start_time = time.time()
    cdef int n_reads = len(reads)
    cdef np.ndarray[np.uint64_t] numeric_read, reverse_read
    cdef np.ndarray[np.uint64_t] kmers, short_kmers, kmer_hashes
    cdef np.ndarray[np.uint64_t] power_array = np.power(4, np.arange(0, k, dtype=np.uint64))
    cdef np.ndarray[np.uint64_t] power_array_short = np.power(4, np.arange(0, k_short, dtype=np.uint64))
    cdef np.ndarray[np.uint64_t] node_hits
    cdef np.ndarray[np.uint64_t] ref_offsets_hits
    cdef np.ndarray[np.uint64_t] read_offsets_hits
    cdef np.ndarray[np.uint64_t] frequencies
    cdef np.ndarray[np.uint64_t, ndim=2] index_lookup_result
    cdef list forward_and_reverse_chains
    cdef float chain_score
    cdef unsigned long ref_start, ref_end
    cdef unsigned long approx_read_length = 150
    cdef unsigned long[:] best_chain_kmers
    cdef int l, n_hits
    #cdef np.ndarray[np.int64_t] reference_kmers = ref_kmers.reference_kmers
    cdef float best_score = 0
    cdef np.ndarray[np.uint16_t] node_counts = np.zeros(max_node_id, dtype=np.uint16)

    cdef unsigned long[:] mapping_positions = np.zeros(n_reads, dtype=np.uint64)
    cdef int n_chains
    cdef int n_unique_short_kmers
    cdef set short_kmers_set
    cdef int u
    cdef float chain_score_int
    cdef unsigned long best_chain_ref_position = 0
    cdef unsigned long approx_read_position

    cdef int[:] short_kmers_array_set = np.zeros(100000007, dtype=np.int32)

    logging.info("Mapping %d reads" % n_reads)

    prev_time = time.time()
    cdef int read_number = -1
    for read in reads:
        if read_number % 20000 == 0:
            logging.info("%d reads processed in %.5f sec" % (read_number, time.time() - prev_time))
            prev_time = time.time()
        read_number += 1

        numeric_read = letter_sequence_to_numeric(np.array(list(read), dtype="|S1").view(np.int8))
        reverse_read = numeric_read[::-1]

        for l in range(2):
            if l == 0:
               kmers = get_kmers(numeric_read, power_array)
               #short_kmers = get_kmers(numeric_read, power_array_short)
               #short_kmers = get_spaced_kmers(numeric_read, power_array_short, k_short)
            else:
               kmers = get_kmers(reverse_read, power_array)
               #short_kmers = get_kmers(reverse_read, power_array_short)
               #short_kmers = get_spaced_kmers(reverse_read, power_array_short, k_short)

            #logging.info(kmers.dtype)
            #logging.info(kmers[0].dtype)
            #short_kmers_set = set(short_kmers)
            #n_unique_short_kmers = len(short_kmers_set)

            index_lookup_result = kmer_index.get(kmers)
            n_hits = index_lookup_result.shape[1]
            if n_hits == 0:
                continue

            node_hits = index_lookup_result[0,:]
            ref_offsets_hits = index_lookup_result[1,:]
            read_offsets_hits = index_lookup_result[2,:]
            frequencies = index_lookup_result[3,:]
            allele_frequencies = index_lookup_result[4,:]

            chains = chain(ref_offsets_hits, read_offsets_hits, node_hits, kmers, frequencies, allele_frequencies)

            n_chains = len(chains)

            for c in range(n_chains):
                ref_start = chains[c][0]
                chain_score_int = chains[c][2]

                if chain_score_int >= best_score:
                    best_score = chain_score_int
                    best_chain = chains[c]
                    best_chain_ref_position = best_chain[0]


        if best_score < 0: # * 150 / 15:  #  * (150 - k_short):
            continue

        best_score = 0.0
        if best_chain is None:
            continue

        # Lookup in reverse index
        best_chain_kmers = best_chain[3]




        """
        best_chain_kmers = best_chain[3]
        rev_kmers, rev_positions, rev_nodes = reverse_index.get_all_between(best_chain_ref_position, best_chain_ref_position + approx_read_length - 31)
        nodes_increased = set()
        got_hit = False
        #nodes_to_increase = []
        for l in range(rev_kmers.shape[0]):
            # Look for match of this kmer in the read
            approx_read_position = rev_positions[l] - best_chain_ref_position
            #assert rev_positions[l] >= best_chain_ref_position and rev_positions[l] < best_chain_ref_position + approx_read_length, "Assert failed on interval %d-%d" % (best_chain_ref_position, best_chain_ref_position + approx_read_length)
            #for c in range(approx_read_position-1, approx_read_position+2):
            #for c in range(approx_read_position-10, approx_read_position+10):
            for c in range(0, len(best_chain_kmers)):
                if c < 0 or c >= len(best_chain_kmers):
                    continue
                if (<np.uint64_t> best_chain_kmers[c]) == (<np.uint64_t> rev_kmers[l]):
                    # Never increase same node more than once for the same read
                    if rev_nodes[l] not in nodes_increased:
                        node_counts[rev_nodes[l]] += 1
                        nodes_increased.add(rev_nodes[l])
                        #nodes_to_increase.append(rev_nodes[l])

                    got_hit = True
                    if rev_nodes[l] == 64088:
                        logging.info("Read: %s, mapped to position %d" % (read, best_chain_ref_position))
                        logging.info("Reverse lookup: %s, %s, %s" % (rev_kmers, rev_positions, rev_nodes))
                        logging.info("    Got a match! Reverse lookup %d, read position: %d" % (l, c))
                        logging.info("Read kmers: %s" % (list(best_chain_kmers)))
                        logging.info("Kmer at position %d: %s" % (c, best_chain_kmers[c]))
                        logging.info("Rev kmers at position %d: %s" % (l, rev_kmers[l]))
                        logging.info("Type: %s" % rev_kmers[l].dtype)
                        logging.info("Type: %s" % type(best_chain_kmers[c]))
                        logging.info(rev_kmers[l] == best_chain_kmers[c])
                        logging.info("%s == %s" % (rev_kmers[l], best_chain_kmers[c]))
                    #break

        """
        """
        if abs(len(set(nodes_to_increase)) - len(set(best_chain[1]))) > 4 and best_chain_ref_position == 2910192 and False:
            logging.info("====== Read pos %d, %s ====" % (best_chain_ref_position, read))
            logging.info("NOdes to increase:      %s" % sorted(list(set(nodes_to_increase))))
            logging.info("NOdes matched in index: %s" % sorted(list(set(best_chain[1]))))
        """

        """
        # Increase node counts
        for l in range(best_chain[1].shape[0]):
            node_counts[best_chain[1][l]] += 1
        """

        mapping_positions[read_number] = best_chain[0]

    logging.info("Time spent: %.4f" % (time.time() - start_time))
    return mapping_positions, node_counts




