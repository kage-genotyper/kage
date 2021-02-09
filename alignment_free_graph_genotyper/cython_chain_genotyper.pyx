import logging
import numpy as np
cimport numpy as np
cimport cython
import time


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list chain(np.ndarray[np.int64_t] ref_offsets, np.ndarray[np.int64_t] read_offsets, np.ndarray[np.int64_t] nodes, np.ndarray[np.int64_t] kmers):

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
            score = np.unique(read_offsets[current_start:i]).shape[0]
            chains.append([potential_chain_start_positions[current_start], nodes[current_start:i], score, kmers])
            current_start = i
        prev_position = potential_chain_start_positions[i]
    score = np.unique(read_offsets[current_start:]).shape[0]
    chains.append([potential_chain_start_positions[current_start], nodes[current_start:], score, kmers])

    return chains


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
def np_letter_sequence_to_numeric(letter_sequence):
    return letter_sequence_to_numeric(letter_sequence.astype("|S1").view(np.int8)).astype(np.uint8)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int64_t] get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
#def get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
    return np.convolve(numeric_read, power_array, mode='valid')



@cython.boundscheck(False)
@cython.wraparound(False)
def run(reads,
        np.int64_t[:] hashes_to_index,
        np.uint32_t[:] n_kmers,
        np.uint32_t[:] nodes,
        np.uint64_t[:] ref_offsets,
        np.uint64_t[:] index_kmers,
        np.uint16_t[:] index_frequencies,
        int modulo,
        int max_node_id,
        int k,
        reference_index,
        int max_index_lookup_frequency,
        reads_are_numeric=False,
        reference_index_scoring=None
        ):

    logging.info("Hash modulo is %d. Max index lookup frequency is %d" % (modulo, max_index_lookup_frequency))


    logging.info("k=%d" % k)
    # Reference index
    cdef unsigned int[:] reference_index_position_to_index = reference_index.ref_position_to_index
    cdef unsigned long[:] reference_index_kmers = reference_index.kmers
    cdef np.ndarray[np.uint32_t] reference_index_nodes = reference_index.nodes
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

    cdef int reads_are_numeric_flag = 0
    if reads_are_numeric:
        reads_are_numeric_flag = 1


    logging.info("Starting cython chaining. N reads: %d" % len(reads))
    prev_time = time.time()

    cdef int read_index

    for read_index in range(len(reads)):
        read = reads[read_index]
        got_index_hits = 0
        if read_number % 100000 == 0:
            logging.info("%d reads processed in %.5f sec. N total chains so far: %d" % (read_number, time.time() - prev_time, n_total_chains))
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
                if do_scoring == 1:
                    short_kmers = get_kmers(numeric_read, power_array_short)
            else:
                kmers = get_kmers(reverse_read, power_array)
                if do_scoring == 1:
                    short_kmers = get_kmers(reverse_read, power_array_short)

            if do_scoring == 1:
                for c in range(short_kmers.shape[0]):
                    short_kmer_set_index[short_kmers[c] % modulo] = 1

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
                    found_nodes[counter] = nodes[index_position + j]
                    found_ref_offsets[counter] = ref_offsets[index_position + j]
                    found_read_offsets[counter] = i
                    counter += 1

            #print(found_nodes)
            #chains = chain(found_ref_offsets, found_read_offsets, found_nodes, kmers)


            # Do the chaining
            potential_chain_start_positions = found_ref_offsets - found_read_offsets
            sorting = np.argsort(potential_chain_start_positions)
            found_read_offsets = found_read_offsets[sorting]
            #found_nodes = found_nodes[sorting]
            potential_chain_start_positions = potential_chain_start_positions[sorting]

            current_start = 0
            prev_position = potential_chain_start_positions[0]
            read_offsets_given_score = set()
            score = 1
            chains = []
            read_offsets_given_score.add(found_read_offsets[0])
            for i in range(1, potential_chain_start_positions.shape[0]):
                if potential_chain_start_positions[i] >= prev_position + 2:
                    #score = np.unique(found_read_offsets[current_start:i]).shape[0]
                    #score = i - current_start  #found_read_offsets[current_start:i].shape[0]
                    chains.append([potential_chain_start_positions[current_start], None, score, kmers])
                    current_start = i
                    score = 0
                    read_offsets_given_score = set()
                prev_position = potential_chain_start_positions[i]

                if found_read_offsets[i] not in read_offsets_given_score:
                    score += 1
                    read_offsets_given_score.add(found_read_offsets[i])

            #score = np.unique(found_read_offsets[current_start:]).shape[0]
            #score = found_read_offsets.shape[0] - current_start
            chains.append([potential_chain_start_positions[current_start], None, score, kmers])

            # Score chains
            if do_scoring == 1:
                for c in range(len(chains)):
                    ##short_kmers_set = chains[c][4]
                    ref_start = chains[c][0] - 5
                    ref_end = ref_start + approx_read_length + 5 - k_short

                    # Get kmers from graph in this area
                    #local_reference_kmers = reference_index_scoring.get_between(ref_start, ref_end)
                    #local_reference_kmers = reference_index_scoring_kmers[reference_index_scoring_position_to_index[ref_start]:reference_index_scoring_position_to_index[ref_end]]
                    local_reference_kmers = reference_index_scoring_kmers[ref_start:ref_end]  #reference_index_scoring_position_to_index[ref_start]:reference_index_scoring_position_to_index[ref_end]]
                    score = 0
                    for i in range(local_reference_kmers.shape[0]):
                        if short_kmer_set_index[local_reference_kmers[i] % modulo] == 1:
                            score += 1
                            # don't count same kmer twice:
                            short_kmer_set_index[local_reference_kmers[i] % modulo] = 0

                    chains[c][2] = score  # len(short_kmers_set.intersection(local_reference_kmers))

                    for c in range(short_kmers.shape[0]):
                        short_kmer_set_index[short_kmers[c] % modulo] = 1

                for c in range(short_kmers.shape[0]):
                    short_kmer_set_index[short_kmers[c] % modulo] = 0

            forward_and_reverse_chains.extend(chains)


            #n_total_chains += len(chains)

            #forward_and_reverse_chains.extend(chains)
            #print(chains)

        #if len(forward_and_reverse_chains) == 0:
        #continue




        # Find best chain
        best_chain_kmers = None
        best_chain = None
        for c in range(len(forward_and_reverse_chains)):
            if forward_and_reverse_chains[c][2] >= best_score:
                best_score = forward_and_reverse_chains[c][2]
                best_chain = forward_and_reverse_chains[c]

        if best_chain is None:
            best_score = 0
            continue

        chain_positions[read_number] = best_chain[0]





        # Align nodes in area of best chain to the kmers (a reverse lookup)
        # Find ref nodes within read area

        # Iterate all nodes, look for SNPs
        best_chain_kmers = best_chain[3]
        best_chain_ref_pos = best_chain[0]

        #best_chain_kmers_set = set(best_chain_kmers)

        # Make a set index for kmers
        for c in range(best_chain_kmers.shape[0]):
            kmer_set_index[best_chain_kmers[c] % modulo] = 1

        reference_kmers_index_start = reference_index_position_to_index[best_chain[0]-10]
        reference_kmers_index_end = reference_index_position_to_index[best_chain[0] + 150 + 10]
        nodes_added = set()

        for c in range(reference_kmers_index_start, reference_kmers_index_end):
            reference_kmer = reference_index_kmers[c]
            current_node = reference_index_nodes[c]

            if kmer_set_index[reference_kmer % modulo] == 1 and current_node not in nodes_added:
                node_counts[current_node] += 1.0
                nodes_added.add(current_node)

                #if current_node == 10702 or current_node == 10703:
                #if current_node == 10947 or current_node == 10950:
                #    logging.info("\nMatch from read %d against node %d on kmer %d. Sequence is %s" % (read_number, current_node, reference_kmer, read))

        # Reset set index
        for c in range(best_chain_kmers.shape[0]):
            kmer_set_index[best_chain_kmers[c] % modulo] = 0



        best_score = 0

    return chain_positions, node_counts






