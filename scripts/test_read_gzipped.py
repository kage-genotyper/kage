from isal import igzip
import bionumpy as bnp
import sys
import time


if __name__ == "__main__":
    file = sys.argv[1]

    t0 = time.perf_counter()
    tstart = time.perf_counter()
    f = bnp.open(file)
    for chunk in f.read_chunks(10000000):
        print(time.perf_counter()-t0)
        t0 = time.perf_counter()

    print("Total", time.perf_counter()-tstart)

