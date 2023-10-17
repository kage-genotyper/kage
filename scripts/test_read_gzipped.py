from isal import igzip
import bionumpy as bnp
import sys
import time
from kmer_mapper.util import open_file


def test_bnp(file):
    t0 = time.perf_counter()
    tstart = time.perf_counter()
    #f = bnp.open(file)
    f = open_file(file)
    out = bnp.open("out.fq.gz", "w")
    for i, chunk in enumerate(f.read_chunks(10000000)):
        time_spent = time.perf_counter()-t0
        print(i, time_spent)
        if time_spent > 0.2:
            #out.write(chunk)
            print("Wrote!")

        t0 = time.perf_counter()

    print("Total", time.perf_counter() - tstart)


def test_isal(file):
    t0 = time.perf_counter()
    tstart = time.perf_counter()
    with igzip.open(file, "rb") as f:
        i = 0
        while f.read(10000000):
            print(i, time.perf_counter() - t0)
            t0 = time.perf_counter()
            i += 1

    print("Total", time.perf_counter() - tstart)



if __name__ == "__main__":
    file = sys.argv[1]
    #test_isal(file)
    test_bnp(file)


