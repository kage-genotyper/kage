import numpy as np
import awkward as ak
import numba



@numba.jit(nopython=True)
def some_func(ak_array):
    j = 0
    for i in range(len(ak_array)):
        for k in range(len(ak_array[i])):

            j = j + 1

    return j


a = ak.Array([[[1, 2, 3], [1, 2]], [[1, 2]]])
print(some_func(a))