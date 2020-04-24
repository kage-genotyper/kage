import logging
import numpy as np
cimport numpy as np
cimport cython
import time

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list chain(long[:] ref_offsets, np.ndarray[np.int64_t] read_offsets, np.ndarray[np.int64_t] nodes):

    cdef np.ndarray[np.int64_t] potential_chain_start_positions = ref_offsets - read_offsets
    cdef int i, start, end


    cdef long[:] sorting = np.argsort(potential_chain_start_positions)
    #ref_offsets = ref_offsets[sorting]
    read_offsets = read_offsets[sorting]
    nodes = nodes[sorting]
    potential_chain_start_positions = potential_chain_start_positions[sorting]


    cdef list chains = []
    cdef int score
    cdef int current_start = 0
    cdef int prev_position = potential_chain_start_positions[0]
    for i in range(1, potential_chain_start_positions.shape[0]):
        if potential_chain_start_positions[i] >= prev_position + 2:
            score = 0  #np.unique(read_offsets[current_start:i]).shape[0]
            chains.append([potential_chain_start_positions[current_start], nodes[current_start:i], score])
            current_start = i
        prev_position = potential_chain_start_positions[i]
    score = 0  # np.unique(read_offsets[current_start:]).shape[0]
    chains.append([potential_chain_start_positions[current_start], nodes[current_start:], score])

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
          np.ndarray[np.int64_t] hashes_to_index,
          np.ndarray[np.uint32_t] n_kmers,
          np.uint32_t[:] nodes,
          np.uint32_t[:] ref_offsets,
          np.uint64_t[:] index_kmers,
          np.uint16_t[:] index_frequencies,
          int modulo,
          np.ndarray[np.int64_t] reference_kmers,
          int max_node_id,
          int k_short,
          int k
          ):

    logging.info("Using small k %d" % k_short)
    cdef np.ndarray[np.int64_t] node_counts = np.zeros(max_node_id+1, dtype=np.int64)
    cdef np.ndarray[np.int64_t] power_array = np.power(4, np.arange(0, k))
    cdef np.ndarray[np.int64_t] power_array_short = np.power(4, np.arange(0, k_short))

    if type(reads_file_name) == list:
        file = reads_file_name
    else:
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
    #cdef np.ndarray[np.int64_t] local_reference_kmers
    cdef long[:] local_reference_kmers
    cdef set short_kmers_set
    cdef float best_score = 0
    cdef list best_chain
    cdef read_number = -1
    cdef np.ndarray[np.int64_t] chain_positions = np.zeros(len(file)//2, dtype=np.int64)
    #cdef np.ndarray[np.int8_t] short_kmers_index = np.zeros(4**k_short, dtype=np.int8)


    logging.info("Starting cython chaining.")
    prev_time = time.time()
    for line in file:
        if line.startswith(">"):
            continue
        if read_number % 1000 == 0:
            logging.info("%d reads processed in %.5f sec" % (read_number, time.time() - prev_time))
            prev_time = time.time()

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
            #short_kmers_set = set(short_kmers[::k_short])
            # Create "set" index for short kmers
            #short_kmers_index[short_kmers] = 1
            #for i in range(short_kmers.shape[0]):
            #short_kmers_index[short_kmers[i]] = 1

            kmer_hashes = np.mod(kmers, modulo)
            n = kmers.shape[0]
            n_total_hits = 0

            # First find number of hits
            for i in range(n):
                hash = kmer_hashes[i]
                if hash == 0:
                    continue
                n_local_hits = n_kmers[hash]
                if n_local_hits > 10000:
                    #logging.warning("%d kmer hits for kmer %d" % (n_local_hits, kmers[i]))
                    continue

                index_position = hashes_to_index[hash]
                for j in range(n_local_hits):
                    # Check that this entry actually matches the kmer, sometimes it will not due to collision
                    #logging.info("Checking index position %d. index_kmers len: %d" % (index_position + j, len(index_kmers)))
                    if index_kmers[index_position+j] != kmers[i]:
                        continue

                    if index_frequencies[index_position+j] > 50:
                        continue
                    n_total_hits += 1

            if n_total_hits == 0:
                continue

            found_nodes = np.zeros(n_total_hits, dtype=np.int)
            found_ref_offsets = np.zeros(n_total_hits, dtype=np.int)
            found_read_offsets = np.zeros(n_total_hits, dtype=np.int)

            # Get the actual hits
            counter = 0

            for i in range(n):
                hash = kmer_hashes[i]
                if hash == 0:
                    continue

                index_position = hashes_to_index[hash]
                n_local_hits = n_kmers[hash]

                if n_local_hits == 0:
                    continue

                if n_local_hits > 10000:
                    #logging.warning("%d kmer hits for kmer %d (skipping 2)" % (n_local_hits, kmers[i]))
                    continue

                for j in range(n_local_hits):
                    if index_kmers[index_position+j] != kmers[i]:
                        continue
                    if index_frequencies[index_position+j] > 50:
                        continue
                    found_nodes[counter] = nodes[index_position+j]
                    found_ref_offsets[counter] = ref_offsets[index_position+j]
                    found_read_offsets[counter] = i
                    counter += 1


            #print(found_nodes)
            chains = chain(found_ref_offsets, found_read_offsets, found_nodes)

            # score chains

            for c in range(len(chains)):
                ref_start = chains[c][0]
                ref_end = ref_start + approx_read_length
                #local_reference_kmers = reference_kmers[ref_start:ref_end-k_short]
                #short_kmers_set.intersection(reference_kmers[ref_start:ref_end-k_short])
                chains[c][2] = len(short_kmers_set.intersection(reference_kmers[ref_start:ref_end-k_short]))
                #chains[c][2] = chain_score


            forward_and_reverse_chains.extend(chains)
            #print(chains)

        #if len(forward_and_reverse_chains) == 0:
        #continue

        # Find best chain
        best_score = 0
        for c in range(len(forward_and_reverse_chains)):
            if forward_and_reverse_chains[c][2] >= best_score:
                best_score = forward_and_reverse_chains[c][2]
                best_chain = forward_and_reverse_chains[c]

        chain_positions[read_number] = best_chain[0]

        for c in range(best_chain[1].shape[0]):
            node_counts[best_chain[1][c]] += 1

        #print("%d\t%d\t%s" % (read_number, best_chain[0], best_chain))

    return chain_positions, node_counts






