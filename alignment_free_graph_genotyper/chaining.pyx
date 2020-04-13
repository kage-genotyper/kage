import numpy as np
cimport numpy as np
cimport cython



@cython.boundscheck(False)
@cython.wraparound(False)
def chain(np.ndarray[np.int64_t] ref_offsets, np.ndarray[np.int64_t] read_offsets, np.ndarray[np.int64_t] nodes):
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
def chain_with_score(np.ndarray[np.int64_t] ref_offsets, np.ndarray[np.int64_t] read_offsets, np.ndarray[np.int64_t] nodes,
               np.ndarray[np.int64_t] reference_kmers,
               np.ndarray[np.int64_t] short_read_kmers):

    short_read_kmers_set = set(short_read_kmers)
    potential_chain_start_positions = ref_offsets - read_offsets
    sorting = np.argsort(potential_chain_start_positions)
    ref_offsets = ref_offsets[sorting]
    nodes = nodes[sorting]
    potential_chain_start_positions = potential_chain_start_positions[sorting]

    cdef read_length = 150
    cdef short_kmer_length = 16
    cdef chains = []
    cdef float best_score = 0
    cdef int best_chain_position = 0
    cdef np.ndarray[np.int64_t] best_nodes = np.zeros(1, dtype=np.int)

    cdef int i, start, end
    cdef float score
    cdef int current_start = 0
    cdef int chain_start
    cdef int prev_position = potential_chain_start_positions[0]

    for i in range(1, len(potential_chain_start_positions)):
        if potential_chain_start_positions[i] >= prev_position + 2:
            chain_start = potential_chain_start_positions[current_start]
            #reference_kmers_set = set(reference_kmers[chain_start:chain_start+150])
            score = len(short_read_kmers_set.intersection(reference_kmers[chain_start:chain_start+150-short_kmer_length])) / len(short_read_kmers_set)
            #score = 0.1
            chains.append([chain_start, nodes[current_start:i], score])
            if score > best_score and False:
                best_score = score
                best_chain_position = chain_start
                best_nodes = nodes[current_start:i]

            current_start = i

        prev_position = potential_chain_start_positions[i]

    # Add last chain
    chain_start = potential_chain_start_positions[current_start]
    score = len(short_read_kmers_set.intersection(reference_kmers[chain_start:chain_start+150-short_kmer_length])) / len(short_read_kmers_set)
    chains.append([chain_start, nodes[current_start:], 0])

    return chains
    #return best_chain_position, best_nodes, best_score
