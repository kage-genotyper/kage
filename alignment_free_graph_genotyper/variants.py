import logging

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

class VariantGenotype:
    def __init__(self, position, ref_sequence, variant_sequence, genotype=None, type="", vcf_line=None):
        self.position = position
        self.ref_sequence = ref_sequence
        self.variant_sequence = variant_sequence
        self.genotype = genotype
        self.vcf_line = vcf_line
        if self.genotype == "1|0":
            self.genotype = "0|1"

        self.type = type

    def id(self):
        return (self.position, self.ref_sequence, self.variant_sequence)

    def __str__(self):
        return "%d %s/%s %s %s" % (self.position, self.ref_sequence, self.variant_sequence, self.genotype, self.type)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if self.id() == other.id():
            if self.genotype == other.genotype:
                return True
        else:
            logging.error("ID mismatch: %s != %s" % (self.id(), other.id()))

        return False

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
    def from_vcf_line(cls, line):
        l = line.split()
        position = int(l[1])
        ref_sequence = l[3].lower()
        variant_sequence = l[4].lower()

        if len(l) >= 10:
            genotype = l[9].split(":")[0].replace("/", "|")
        else:
            genotype = ""

        return cls(position, ref_sequence, variant_sequence, genotype, get_variant_type(line), line)


class GenotypeCalls:
    def __init__(self, variant_genotypes, skip_index=False):
        self.variant_genotypes = variant_genotypes
        self._index = {}
        if not skip_index:
            self.make_index()

    def get_chunks(self, chunk_size=5000):
        out = []
        i = 0
        for variant_number, variant in enumerate(self.variant_genotypes):
            if variant_number % 1000 == 0:
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

    @classmethod
    def from_vcf(cls, vcf_file_name, skip_index=False, limit_to_n_lines=None, make_generator=False):
        logging.info("Reading variants from file")
        variant_genotypes = []

        f = open(vcf_file_name)

        if make_generator:
            logging.info("Returning variant generator")
            return cls((VariantGenotype.from_vcf_line(line) for line in f if not line.startswith("#")), skip_index=skip_index)

        for i, line in enumerate(f):
            if line.startswith("#"):
                continue

            if i % 10000 == 0:
                logging.info("Read %d variants from file" % i)

            if limit_to_n_lines is not None and i >= limit_to_n_lines:
                logging.warning("Limited to %d lines" % limit_to_n_lines)
                break

            variant_genotypes.append(VariantGenotype.from_vcf_line(line))

        return cls(variant_genotypes, skip_index)

