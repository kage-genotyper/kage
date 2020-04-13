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
