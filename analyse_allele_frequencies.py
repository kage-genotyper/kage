import gzip
import sys
import logging
logging.basicConfig(level=logging.INFO)

f = gzip.open(sys.argv[1])

extra_header = \
"""##INFO=<ID=AF_HOMO_REF,Number=A,Type=Float,Description="Allele frequency homozygous ref, in the range (0,1)">
##INFO=<ID=AF_HOMO_ALT,Number=A,Type=Float,Description="Allele frequency homozygous alt, in the range (0,1)">
##INFO=<ID=AF_HETERO,Number=A,Type=Float,Description="Allele frequency heterozygous, in the range (0,1)">"""

header_inserted = False
for line in f:
    line = line.decode("utf-8")
    if line.startswith("##INFO") and not header_inserted:
        print(extra_header)
        header_inserted = True

    if line.startswith("#"):
        print(line.strip())
        continue


    n_hetero = 0
    n_homo_ref = 0
    n_homo_alt = 0

    l = line.split()
    n_tot = len(l[9:])
    for genotype in l[9:]:
        if genotype == "0|0":
            n_homo_ref += 1
        elif genotype == "0|1" or genotype == "1|0":
            n_hetero += 1
        elif genotype == "1|1":
            n_homo_alt += 1
        else:
            raise Exception("Invalid genotype %s" % genotype)

    allele_freq = float(l[7].split("AF=")[1].split(";")[0])
    frequency_homo_ref = n_homo_ref / n_tot
    frequency_homo_alt = n_homo_alt / n_tot
    frequency_hetero = n_hetero / n_tot

    expected_homo_ref = (1-allele_freq)**2
    expected_homo_alt = allele_freq**2
    expected_hetero = 1 - expected_homo_ref - expected_homo_alt

    #logging.info("Allele freq: %.4f, homo ref: %.4f (%.4f), homo alt: %.4f (%.4f), hetero: %.4f (%.4f)" % (allele_freq, frequency_homo_ref, expected_homo_ref, frequency_homo_alt, expected_homo_alt, frequency_hetero, expected_hetero))

    l[7] += ";AF_HOMO_REF=%.3f;AF_HOMO_ALT=%.3f;AF_HETERO=%.3f;" % (frequency_homo_ref, frequency_homo_alt, frequency_hetero)
    print('\t'.join(l))

