import logging
import gzip
import io
import time

def get_variant_type(vcf_line):

    l = vcf_line.split()
    if len(l[3]) == len(l[4]):
        return "SNP"
    elif "VT=SNP" in vcf_line:
        return "SNP"
    elif len(l[3]) > len(l[4]):
        return "DELETION"
    elif len(l[3]) < len(l[4]):
        return "INSERTION"
    elif "VT=INDEL" in vcf_line:
        if len(l[3]) > len(l[4]):
            return "DELETION"
        else:
            return "INSERTION"
    else:
        return ""
        raise Exception("Unsupported variant type on line %s" % vcf_line)


class TruthRegions:
    def __init__(self, file_name):
        self.regions = []
        f = open(file_name)

        for line in f:
            l = line.split()
            start = int(l[1])
            end = int(l[2])

            self.regions.append((start, end))

    def is_inside_regions(self, position):
        for region in self.regions:
            if position >= region[0] and position < region[1]:
                return True
        return False


class VcfVariant:
    def __init__(self, chromosome, position, ref_sequence, variant_sequence, genotype=None, type="", vcf_line=None, vcf_line_number=None):
        self.chromosome = chromosome
        self.position = position
        self.ref_sequence = ref_sequence
        self.variant_sequence = variant_sequence
        self.genotype = genotype
        self.vcf_line = vcf_line
        if self.genotype == "1|0":
            self.genotype = "0|1"

        self.type = type
        self.vcf_line_number = vcf_line_number


    def get_genotype(self):
        return self.genotype.replace("|", "/").replace("1/0", "0/1")

    def copy(self):
        return VcfVariant(self.chromosome, self.position, self.ref_sequence, self.variant_sequence, self.genotype, self.type, self.vcf_line, self.vcf_line_number)

    def set_genotype(self, genotype):
        assert genotype in ["0|0", "0/0", "0|1", "0/1", "1/1", "1|1"], "Invalid genotype %s" % genotype
        self.genotype = genotype

    def id(self):
        return (self.chromosome, self.position, self.ref_sequence, self.variant_sequence)

    def get_vcf_line(self):
        return "%d\t%d\t.\t%s\t%s\t.\tPASS\t.\tGT\t%s\n"  % (self.chromosome, self.position, self.ref_sequence, self.variant_sequence, self.genotype if self.genotype is not None else ".")

    def __str__(self):
        return "chr%d:%d %s/%s %s %s" % (self.chromosome, self.position, self.ref_sequence, self.variant_sequence, self.genotype, self.type)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if self.id() == other.id():
            if self.genotype == other.genotype:
                return True
        else:
            logging.error("ID mismatch: %s != %s" % (self.id(), other.id()))

        return False

    def get_deleted_sequence(self):
        if self.type == "DELETION":
            return self.ref_sequence[1:]

        raise Exception("Variant %s is not a deletion" % self)

    def get_variant_sequence(self):
        if self.type == "DELETION":
            return ""
        elif self.type == "INSERTION":
            return self.variant_sequence[1:]
        else:
            return self.variant_sequence


    def get_variant_allele_frequency(self):
        return float(self.vcf_line.split("AF=")[1].split(";")[0])

    def get_reference_allele_frequency(self):
        return 1 - self.get_variant_allele_frequency()

    def get_individuals_and_numeric_genotypes(self):
        # genotypes are 1, 2, 3 for homo ref, homo alt and hetero
        for individual_id, genotype_string in enumerate(self.vcf_line.split()[9:]):
            genotype_string = genotype_string.split(":")[0].replace("/", "|")
            if genotype_string == "0|0":
                numeric = 1
            elif genotype_string == "1|1":
                numeric = 2
            elif genotype_string == "0|1" or genotype_string == "1|0":
                numeric = 3
            else:
                logging.error("Could not parse genotype string %s" % genotype_string)
                raise Exception("Unknown genotype")

            yield (individual_id, numeric)

    @classmethod
    def from_vcf_line(cls, line, vcf_line_number=None):
        l = line.split()
        chromosome = l[0]
        if chromosome == "X":
            chromosome = 23
        elif chromosome == "Y":
            chromosome = 24

        chromosome = int(chromosome)
        position = int(l[1])
        ref_sequence = l[3].lower()
        variant_sequence = l[4].lower()

        if len(l) >= 10:
            genotype = l[9].split(":")[0].replace("/", "|")
        else:
            genotype = ""

        return cls(chromosome, position, ref_sequence, variant_sequence, genotype, get_variant_type(line), line, vcf_line_number=vcf_line_number)

    def get_reference_position_before_variant(self):
        # Returns the position of the last base pair before the variant starts (will be end of node before variant)
        # For SNPS, the position is the actual SNP node
        # FOr deletions, the position is one basepair before the deleted sequence
        # For insertion, the position is one baseiapr before the inserted sequence
        # Subtract -1 in the end to make it 0-based
        if self.type == "SNP":
            return self.position - 1 - 1
        else:
            return self.position - 1

    def get_reference_position_after_variant(self):
        # Returns the next reference position after the variant is finished
        start = self.get_reference_position_before_variant()
        if self.type == "SNP":
            return start + 2
        elif self.type == "INSERTION":
            return start + 1
        elif self.type == "DELETION":
            return start + len(self.ref_sequence) - 1 + 1


class VcfVariants:
    def __init__(self, variant_genotypes, skip_index=False, header_lines=""):
        self._header_lines = header_lines
        self.variant_genotypes = variant_genotypes
        self._index = {}
        if not skip_index:
            self.make_index()

    def get_chunks(self, chunk_size=5000):
        out = []
        i = 0
        for variant_number, variant in enumerate(self.variant_genotypes):
            if variant_number % 100000 == 0:
                logging.info("%d variants read" % variant_number)
            out.append(variant)
            i += 1
            if i >= chunk_size and chunk_size > 0:
                yield out
                out = []
                i = 0
        yield out

    def make_index(self):
        logging.info("Making vcf index")
        for variant in self.variant_genotypes:
            self._index[variant.id()] = variant
        logging.info("Done making vcf index")

    def has_variant(self, variant_genotype):
        if variant_genotype.id() in self._index:
            return True

        return False

    def has_variant_genotype(self, variant_genotype):
        if self.has_variant(variant_genotype) and self._index[variant_genotype.id()].genotype == variant_genotype.genotype:
            return True

        return False

    def get(self, variant):
        return self._index[variant.id()]

    def __getitem__(self, item):
        return self.variant_genotypes[item]

    def __len__(self):
        return len(self.variant_genotypes)

    def __iter__(self):
        return self.variant_genotypes.__iter__()

    def __next__(self):
        return self.variant_genotypes.__next__()

    def to_vcf_file(self, file_name, add_individual_to_header="DONOR"):
        logging.info("Writing to file %s" % file_name)
        with open(file_name, "w") as f:
            #f.write(self._header_lines)
            for header_line in self._header_lines.split("\n")[0:-1]:  # last element is empty
                if header_line.startswith("#CHROM"):
                    header_line += "\tDONOR"
                f.writelines([header_line + "\n"])

            for i, variant in enumerate(self):
                if i % 10000 == 0:
                    logging.info("%d variants written to file" % i)

                f.writelines([variant.get_vcf_line()])


    @classmethod
    def from_vcf(cls, vcf_file_name, skip_index=False, limit_to_n_lines=None, make_generator=False, limit_to_chromosome=None):
        logging.info("Reading variants from file")
        variant_genotypes = []

        if limit_to_chromosome is not None:
            if limit_to_chromosome == "X":
                limit_to_chromosome = "23"
            elif limit_to_chromosome == "Y":
                limit_to_chromosome = "24"

        if limit_to_chromosome is not None:
            logging.info("Will only read variants from chromsome %s" % limit_to_chromosome)

        f = open(vcf_file_name)
        is_bgzipped = False
        if vcf_file_name.endswith(".gz"):
            logging.info("Assuming gzipped file")
            is_bgzipped = True
            gz = gzip.open(vcf_file_name)
            f = io.BufferedReader(gz, buffer_size=1000 * 1000 * 2)  # 2 GB buffer size?
            logging.info("Made gz file object")

        if make_generator:
            logging.info("Returning variant generator")
            if is_bgzipped:
                f = (line for line in f if not line.decode("utf-8").startswith("#"))
                return cls((VcfVariant.from_vcf_line(line.decode("utf-8"), vcf_line_number=i) for i, line in enumerate(f) if not line.decode("utf-8").startswith("#")), skip_index=skip_index)
            else:
                f = (line for line in f if not line.startswith("#"))
                return cls((VcfVariant.from_vcf_line(line, vcf_line_number=i) for i, line in enumerate(f)), skip_index=skip_index)

        header_lines = ""
        n_variants_added = 0
        prev_time = time.time()
        variant_number = -1
        for i, line in enumerate(f):
            if is_bgzipped:
                line = line.decode("utf-8")


            if line.startswith("#"):
                header_lines += line
                continue

            variant_number += 1

            if i % 50000 == 0:
                logging.info("Read %d variants from file (time: %.3f). %d variants added" % (i, time.time()-prev_time, n_variants_added))
                prev_time = time.time()

            if limit_to_n_lines is not None and i >= limit_to_n_lines:
                logging.warning("Limited to %d lines" % limit_to_n_lines)
                break


            variant = VcfVariant.from_vcf_line(line, vcf_line_number=variant_number)
            if limit_to_chromosome is not None and variant.chromosome != int(limit_to_chromosome):
                continue

            n_variants_added += 1
            variant_genotypes.append(variant)

        return cls(variant_genotypes, skip_index, header_lines)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return VcfVariants([v.copy() for v in self])

    def compute_similarity_to_other_variants(self, other_variants):
        n_identical = 0
        for variant, other_variant in zip(self, other_variants):
            assert variant.position == other_variant.position
            if variant.genotype == other_variant.genotype:
                n_identical += 1

        return n_identical / len(self)