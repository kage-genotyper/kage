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
    def __init__(self, position, ref_sequence, variant_sequence, genotype, type="", vcf_line=None):
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
    def __init__(self, variant_genotypes):
        self.variant_genotypes = variant_genotypes
        self._index = {}
        self.make_index()

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
        if self.has_variant(variant_genotype) and self._index[variant_genotype.id()].genotyep == variant_genotype.genotype:
            return True

        return False

    def get(self, variant):
        return self._index[variant.id()]

    def __iter__(self):
        return self.variant_genotypes.__iter__()

    def __next__(self):
        return self.variant_genotypes.__next__()

    @classmethod
    def from_vcf(cls, vcf_file_name):
        variant_genotypes = []

        f = open(vcf_file_name)
        for line in f:
            if line.startswith("#"):
                continue
            variant_genotypes.append(VariantGenotype.from_vcf_line(line))

        return cls(variant_genotypes)

