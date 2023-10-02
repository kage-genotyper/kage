import numba
import numpy as np


class ModuloFilter:
    def __init__(self, array):
        self._array = array
        self._modulo = len(array)

    @classmethod
    def empty(cls, modulo):
        return cls(np.zeros(modulo, dtype=bool))

    def add(self, elements):
        self._array[elements % self._modulo] = True

    def __getitem__(self, elements):
        keys = elements % self._modulo
        return self._array[keys]

    def copy(self):
        return ModuloFilter(self._array.copy())

    def getitem_numba(self, elements):
        out = np.zeros_like(elements, dtype=bool)

        @numba.njit
        def _get(out, elements, array):
            modulo = len(array)
            for i in range(len(elements)):
                out[i] = array[elements[i] % modulo]

        _get(out, elements, self._array)
        return out

