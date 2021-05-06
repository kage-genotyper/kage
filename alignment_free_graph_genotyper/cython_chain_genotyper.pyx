import logging
import numpy as np
cimport numpy as np
cimport cython
import time


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int64_t, ndim=1] complement_of_numeric_read(np.ndarray[np.int64_t, ndim=1] numeric_read):
    cdef np.ndarray[np.int64_t] complement = np.zeros(numeric_read.shape[0], dtype=np.int)
    cdef int i = 0
    cdef int base
    for i in range(0, numeric_read.shape[0]):
        base = numeric_read[i]
        if base == 0:
            complement[i] = 2
        elif base == 1:
            complement[i] = 3
        elif base == 2:
            complement[i] = 0
        elif base == 3:
            complement[i] = 1
    return complement

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int64_t, ndim=1] letter_sequence_to_numeric(np.ndarray[np.int8_t, ndim=1] letter_sequence):
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
def np_letter_sequence_to_numeric(letter_sequence):
    return letter_sequence_to_numeric(letter_sequence.astype("|S1").view(np.int8)).astype(np.uint8)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int64_t] get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
    return np.convolve(numeric_read, power_array, mode='valid')



@cython.boundscheck(False)
@cython.wraparound(False)
def run(reads,
        index,
        int max_node_id,
        int k,
        reference_index,
        int max_index_lookup_frequency,
        reads_are_numeric=False,
        reference_index_scoring=None,
        skip_chaining=False,
        scale_by_frequency=False
        ):

    if skip_chaining:
        logging.warning("Will not do any chaining.")

    cdef np.int64_t[:] hashes_to_index = index._hashes_to_index
    cdef np.uint32_t[:] n_kmers = index._n_kmers
    cdef np.uint32_t[:] nodes = index._nodes
    cdef np.uint64_t[:] ref_offsets = index._ref_offsets
    cdef np.uint64_t[:] index_kmers = index._kmers
    cdef np.uint16_t[:] index_frequencies = index._frequencies
    cdef int modulo = index._modulo
    logging.info("Hash modulo is %d. Max index lookup frequency is %d. k=%d" % (modulo, max_index_lookup_frequency, k))

    logging.info("k=%d" % k)
    # Reference index

    cdef unsigned int[:] reference_index_position_to_index
    cdef unsigned long[:] reference_index_kmers
    cdef np.ndarray[np.uint32_t] reference_index_nodes
    if reference_index is not None:
        reference_index_position_to_index = reference_index.ref_position_to_index
        reference_index_kmers = reference_index.kmers
        reference_index_nodes = reference_index.nodes


    cdef unsigned int reference_kmers_index_start
    cdef unsigned int reference_kmers_index_end

    # Reference index scoring
    #cdef unsigned int[:] reference_index_scoring_position_to_index = reference_index_scoring.ref_position_to_index
    cdef unsigned int[:] reference_index_scoring_kmers
    cdef int do_scoring = 0
    if reference_index_scoring is not None:
        reference_index_scoring_kmers = reference_index_scoring.kmers
        do_scoring = 1
    else:
        logging.warning("Skipping scoring against reference kmers")

    if scale_by_frequency:
        logging.info("Will scale counts by frequency")


    cdef np.ndarray[np.float_t] node_counts = np.zeros(max_node_id+1, dtype=np.float)
    cdef np.ndarray[np.int64_t] power_array = np.power(4, np.arange(0, k))
    cdef int k_short = 15;
    cdef np.ndarray[np.int64_t] power_array_short = np.power(4, np.arange(0, k_short))
    cdef np.ndarray[np.uint8_t] kmer_set_index = np.zeros(modulo, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t] short_kmer_set_index = np.zeros(modulo, dtype=np.uint8)
    cdef long reference_kmer

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
    cdef np.ndarray[np.int64_t] found_frequencies

    cdef int l, c, r, a
    cdef int counter = 0
    cdef int n_local_hits, j
    cdef list forward_and_reverse_chains
    cdef float chain_score
    cdef long ref_start, ref_end
    cdef int approx_read_length = 150
    #cdef np.ndarray[np.int64_t] local_reference_kmers
    cdef unsigned int[:] local_reference_kmers
    cdef set short_kmers_set
    cdef float best_score = 0
    #cdef list best_chain
    cdef read_number = -1
    cdef np.ndarray[np.int64_t] chain_positions = np.zeros(len(reads), dtype=np.int64)
    cdef np.ndarray[np.uint32_t] ref_nodes_in_read_area = np.zeros(0, dtype=np.uint32)
    cdef np.ndarray[np.int64_t] snp_nodes = np.zeros(0, dtype=np.int64)
    #cdef np.ndarray[np.int8_t] short_kmers_index = np.zeros(4**k_short, dtype=np.int8)
    cdef np.uint64_t[:] reverse_kmers
    cdef np.uint64_t[:] reverse_ref_positions

    cdef long best_chain_ref_pos = 0
    cdef np.uint32_t ref_node, current_node
    cdef np.uint32_t edges_index, n_edges
    cdef np.ndarray[np.int64_t] best_chain_kmers
    cdef int read_pos, reverse_index_index
    cdef int current_node_match
    cdef int got_index_hits = 0
    cdef int n_total_chains = 0

    # Variables used in chaining
    cdef np.ndarray[np.int64_t] potential_chain_start_positions
    cdef int start, end
    cdef long[:] sorting
    cdef int score
    cdef int current_start = 0
    cdef int prev_position
    cdef set read_offsets_given_score
    cdef set nodes_added
    cdef long pos

    cdef int reads_are_numeric_flag = 0
    if reads_are_numeric:
        reads_are_numeric_flag = 1


    logging.info("Starting cython chaining. N reads: %d" % len(reads))
    prev_time = time.time()

    cdef int read_index

    for read_index in range(len(reads)):
        read = reads[read_index]
        got_index_hits = 0
        if read_number % 10000 == 0:
            logging.info("%d reads processed (last 10k processed in %.5f sec). N total chains so far: %d" % (read_number, time.time() - prev_time, n_total_chains))
            prev_time = time.time()

        read_number += 1


        if reads_are_numeric_flag == 0:
            numeric_read = letter_sequence_to_numeric(np.array(list(read), dtype="|S1").view(np.int8))
        else:
            numeric_read = read.astype(np.int64)



        reverse_read = complement_of_numeric_read(numeric_read[::-1])
        forward_and_reverse_chains = []

        for l in range(2):
            if l == 0:
                kmers = get_kmers(numeric_read, power_array)
            else:
                kmers = get_kmers(reverse_read, power_array)

            #short_kmers_set = set(short_kmers)
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
                    if index_kmers[index_position + j] != kmers[i]:
                        continue

                    if index_frequencies[index_position + j] > max_index_lookup_frequency:
                        continue
                    n_total_hits += 1

            if n_total_hits == 0:
                continue


            found_nodes = np.zeros(n_total_hits, dtype=np.int64)
            found_ref_offsets = np.zeros(n_total_hits, dtype=np.int64)
            found_read_offsets = np.zeros(n_total_hits, dtype=np.int64)
            found_frequencies = np.zeros(n_total_hits, dtype=np.int64)

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
                    if index_kmers[index_position + j] != kmers[i]:
                        continue
                    if index_frequencies[index_position + j] > max_index_lookup_frequency:
                        continue
                    #print(kmers[i])
                    found_nodes[counter] = nodes[index_position + j]
                    found_ref_offsets[counter] = ref_offsets[index_position + j]
                    found_read_offsets[counter] = i
                    found_frequencies[counter] = index_frequencies[index_position]
                    counter += 1

            #print(found_nodes)
            #chains = chain(found_ref_offsets, found_read_offsets, found_nodes, kmers)


            # Do the chaining
            if skip_chaining:
                for c in range(found_nodes.shape[0]):
                    if scale_by_frequency:
                        node_counts[found_nodes[c]] += 1.0  / found_frequencies[c]
                    else:
                        node_counts[found_nodes[c]] += 1

    return node_counts


