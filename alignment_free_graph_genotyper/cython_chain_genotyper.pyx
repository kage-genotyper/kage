import logging
import numpy as np
cimport numpy as np
cimport cython
import time


#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef list chain(long[:] ref_offsets, np.ndarray[np.int64_t] read_offsets, np.ndarray[np.int64_t] nodes, np.ndarray[np.int64_t] kmers):

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
            #score = read_offsets[current_start:i].shape[0]
            chains.append([potential_chain_start_positions[current_start], nodes[current_start:i], score, kmers])
            current_start = i
        prev_position = potential_chain_start_positions[i]
    score = np.unique(read_offsets[current_start:]).shape[0]
    #score = read_offsets[current_start:].shape[0]
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
cdef np.ndarray[np.int64_t] get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
#def get_kmers(np.ndarray[np.int64_t] numeric_read, np.ndarray[np.int64_t] power_array):
    return np.convolve(numeric_read, power_array, mode='valid')



@cython.boundscheck(False)
@cython.wraparound(False)
def run(reads,
          np.ndarray[np.int64_t] hashes_to_index,
          np.ndarray[np.uint32_t] n_kmers,
          np.uint32_t[:] nodes,
          np.uint64_t[:] ref_offsets,
          np.uint64_t[:] index_kmers,
          np.uint16_t[:] index_frequencies,
          int modulo,
          int max_node_id,
          int k,
          skip_chaining=False
          ):

    logging.info("Hash modulo is %d" % modulo)

    if skip_chaining:
        logging.info("Will skip chaining completely")

    logging.info("k=%d" % k)

    cdef np.ndarray[np.float_t] node_counts = np.zeros(max_node_id+1, dtype=np.float)
    cdef np.ndarray[np.int64_t] power_array = np.power(4, np.arange(0, k))


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
    cdef long[:] local_reference_kmers
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

    logging.info("Starting cython chaining. N reads: %d" % len(reads))
    prev_time = time.time()
    for read in reads:
        got_index_hits = 0
        if read_number % 50000 == 0:
            logging.info("%d reads processed in %.5f sec" % (read_number, time.time() - prev_time))
            prev_time = time.time()

        read_number += 1


        numeric_read = letter_sequence_to_numeric(np.array(list(read), dtype="|S1").view(np.int8))
        reverse_read = complement_of_numeric_read(numeric_read[::-1])
        forward_and_reverse_chains = []


        for l in range(2):
            #logging.info("l=%d" % l)
            if l == 0:
               kmers = get_kmers(numeric_read, power_array)
            else:
               kmers = get_kmers(reverse_read, power_array)

            kmer_hashes = np.mod(kmers, modulo)
            n = kmers.shape[0]
            n_total_hits = 0

            # First find number of hits
            for i in range(n):
                hash = kmer_hashes[i]
                if hash == 0:
                    continue

                n_local_hits = n_kmers[hash]
                #logging.info("N local hits: %d" % n_local_hits)
                if n_local_hits > 10000:
                    #logging.warning("%d kmer hits for kmer %d" % (n_local_hits, kmers[i]))
                    continue

                index_position = hashes_to_index[hash]
                for j in range(n_local_hits):
                    # Check that this entry actually matches the kmer, sometimes it will not due to collision
                    #logging.info("Checking index position %d. index_kmers len: %d" % (index_position + j, len(index_kmers)))
                    if index_kmers[index_position+j] != kmers[i]:
                        continue

                    if index_frequencies[index_position+j] > 100:
                        continue
                    n_total_hits += 1

            #logging.info("N total hits: %d" % (n_total_hits))

            if n_total_hits == 0:
                #logging.info("0 total hits")
                continue

            got_index_hits = 1


            found_nodes = np.zeros(n_total_hits, dtype=np.int)
            found_ref_offsets = np.zeros(n_total_hits, dtype=np.int)
            found_read_offsets = np.zeros(n_total_hits, dtype=np.int)
            found_frequencies = np.zeros(n_total_hits, dtype=np.int)

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
                    if index_frequencies[index_position+j] > 100:
                        continue
                    found_nodes[counter] = nodes[index_position+j]
                    found_ref_offsets[counter] = ref_offsets[index_position+j]
                    found_read_offsets[counter] = i
                    found_frequencies[counter] = index_frequencies[index_position+j]
                    counter += 1


            #print(found_nodes)
            if not skip_chaining:
                chains = chain(found_ref_offsets, found_read_offsets, found_nodes, kmers)
                forward_and_reverse_chains.extend(chains)
            else:
                for c in range(found_nodes.shape[0]):
                    node_counts[found_nodes[c]] += 1 # / found_frequencies[c]
                continue
            #print(chains)

        #if len(forward_and_reverse_chains) == 0:
        #continue
        #logging.info("Done finding nodes")
        #logging.info("Found %d nodes" % found_nodes.shape[0])

        if skip_chaining:
            continue


        # Find best chain
        best_chain_kmers = None
        best_chain = None
        for c in range(len(forward_and_reverse_chains)):
            if forward_and_reverse_chains[c][2] > best_score:
                best_score = forward_and_reverse_chains[c][2]
                best_chain = forward_and_reverse_chains[c]

        if best_chain is None:
            #logging.warning("No chain found for read %s" % read)
            continue

        chain_positions[read_number] = best_chain[0]
        best_score = 0

        added = set()
        for c in range(best_chain[1].shape[0]):
            node = best_chain[1][c]
            if node not in added:
                node_counts[node] += 1

            added.add(node)


    logging.info("Done with all reads")
    return chain_positions, node_counts






