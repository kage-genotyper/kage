import logging

logging.basicConfig(level=logging.INFO)
import sys
import gzip


def remove_overlapping_indels(vcf_file_name, skip_variants_with_n=True):
    if vcf_file_name.endswith(".gz"):
        is_gzipped = True
        vcf = gzip.open(vcf_file_name)
    else:
        vcf = open(vcf_file_name)

    n_snps = 0
    n_indels_removed = 0

    snp_positions = set()

    # First get position of all SNPs
    for i, line in enumerate(vcf):
        if is_gzipped:
            line = line.decode("utf-8")

        if line.startswith("#"):
            continue

        if i % 100000 == 0:
            logging.info("%d lines processed" % i)

        if "VT=SNP" in line:
            n_snps += 1

            snp_positions.add(int(line.split()[1]))

    if is_gzipped:
        vcf = gzip.open(vcf_file_name)
    else:
        vcf = open(vcf_file_name)

    indel_positions = set()
    for i, line in enumerate(vcf):
        if is_gzipped:
            line = line.decode("utf-8")

        if i % 100000 == 0:
            logging.info("%d lines processed" % i)

        if line.startswith("#"):
            print(line.strip())
            continue

        l = line.split()

        if "N" in l[3] or "N" in l[4]:
            logging.info("Skipped variant %s,%s due to N in sequence" % (l[0], l[1]))
            continue

        if "VT=INDEL" in line:
            pos = int(l[1])
            size = len(l[3]) + 1

            overlapping = False
            for j in range(pos, pos + size):
                if j in snp_positions:
                    overlapping = True

                # also check against indels included so far
                if j in indel_positions:
                    overlapping = True

            for j in range(pos, pos + size):
                indel_positions.add(j)

            if overlapping:
                n_indels_removed += 1
                continue

        print(line.strip())

    logging.info("Indels removed: %d" % n_indels_removed)
