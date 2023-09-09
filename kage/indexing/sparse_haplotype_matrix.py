import logging
from dataclasses import dataclass
import bionumpy as bnp
import numba
import numpy as np
import scipy
import tqdm
from graph_kmer_index.nplist import NpList

from kage.indexing.paths import PathCreator


#class SparseColumnMatrix:
#    pass


#class SparsColumnMatrixRaggedArray:


@dataclass
class SparseHaplotypeMatrix:
    data: scipy.sparse.csc_matrix  # n_variants x n_haplotypes

    def extend(self, other: "SparseHaplotypeMatrix"):
        if self.data is None:
           self.data = other.data
        else:
            self.data = scipy.sparse.vstack([self.data, other.data])

    def __add__(self, other):
        return SparseHaplotypeMatrix(self.data + other.data)

    @classmethod
    def empty(cls):
        return cls(None)

    def to_biallelic(self, n_alleles_per_variant) -> 'SparseHaplotypeMatrix':
        """
        Converts a multiallelic haplotype matrix to biallelic.
        """
        if np.all(n_alleles_per_variant) == 2:
            # already biallelic
            return self

        n_alt_alleles_per_variant = n_alleles_per_variant - 1
        assert len(n_alt_alleles_per_variant) == self.n_variants

        # algorithm idea:
        # create a matrix where rows with more than 2 alleles are duplicated
        # each such row should have 1 allele if it's allele matches the row rank,
        # where rank is 1, 2, 3 ... for the rows
        # first make matrix with duplicate rows, then do modulo
        matrix = self.to_matrix()

        @numba.jit(nopython=True)
        def make_row_indexes(n_alt_alleles_per_variant):
            row_indexes = np.zeros(np.sum(n_alt_alleles_per_variant), dtype=np.int64)
            allele_indexes = np.zeros_like(row_indexes)
            i = 0
            for variant in range(len(n_alt_alleles_per_variant)):
                for alt_allele in range(n_alt_alleles_per_variant[variant]):
                    row_indexes[i] = variant
                    allele_indexes[i] = alt_allele+1
                    i += 1
            return row_indexes, allele_indexes

        row_indexes, allele_indexes = make_row_indexes(n_alt_alleles_per_variant)
        allele_indexes_matrix = np.tile(allele_indexes, (self.n_haplotypes, 1)).T
        new_matrix = matrix[row_indexes, :]
        new_matrix = (new_matrix == allele_indexes_matrix).astype(np.uint8)
        return SparseHaplotypeMatrix.from_nonsparse_matrix(new_matrix)


    def to_multiallelic(self, n_alleles_per_variant) -> 'SparseHaplotypeMatrix':
        """
        Converts a biallelic haplotype matrix to multialellic. Assumes current matrix has two alleles,
        and uses n_alleles_per_variant to group variants.
        """
        columns = []
        for haplotype in range(self.n_haplotypes):
            h = self.get_haplotype(haplotype)
            new_column = PathCreator.convert_biallelic_path_to_multiallelic(
                n_alleles_per_variant,
                h, how="encoding")
            columns.append(new_column)

        return SparseHaplotypeMatrix.from_nonsparse_matrix(np.array(columns).T)

    @classmethod
    def from_nonsparse_matrix(cls, matrix):
        return cls(scipy.sparse.csc_matrix(matrix))

    def to_matrix(self):
        return self.data.toarray()

    @classmethod
    def from_variants_and_haplotypes(cls, variant_ids, haplotype_ids, n_variants, n_haplotypes, values=None):
        """
        variant_ids: np.array of variant ids
        haplotype_ids: np.array of haplotype ids (which haplotypes have the variant allele)
        values: If None, will be filled with ones
        """
        if values is None:
            values = np.ones(len(variant_ids), dtype=np.uint8)
        else:
            values = values.astype(np.uint8)
            assert len(values) == len(variant_ids)
        data = scipy.sparse.csc_matrix((values, (variant_ids, haplotype_ids)),
                                       shape=(n_variants, n_haplotypes))
        return cls(data)

    def get_haplotype(self, haplotype_id):
        return self.data.getcol(haplotype_id).toarray().flatten()

    def get_haplotype_nodes(self, haplotype_id):
        # assuming implicit node conversion
        haplotypes = self.get_haplotype(haplotype_id)
        nodes = np.arange(len(haplotypes)) * 2 + haplotypes
        return nodes

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_individuals(self):
        return self.shape[1] // 2

    @property
    def n_haplotypes(self):
        return self.shape[1]

    @property
    def n_variants(self):
        return self.shape[0]

    def to_file(self, file_name):
        scipy.sparse.save_npz(file_name, self.data)

    @classmethod
    def from_file(cls, file_name):
        data = scipy.sparse.load_npz(file_name)
        return cls(data)


    @classmethod
    def from_vcf2(cls, vcf_file_name, convert_multiallelic_to_biallelic=False) -> 'SparseHaplotypeMatrix':
        # Uses haplotypeencoding
        vcf = bnp.open(vcf_file_name, buffer_type=bnp.io.delimited_buffers.PhasedHaplotypeVCFMatrixBuffer)
        matrix = SparseHaplotypeMatrix.empty()

        n_alleles_per_variant = []

        for i, chunk in enumerate(vcf.read_chunks(min_chunk_size=500000000)):
            genotypes = chunk.genotypes.raw()
            n_haplotypes = genotypes.shape[1]
            variant_ids, haplotypes = np.where(genotypes > 0)
            haplotype_values = genotypes[variant_ids, haplotypes].ravel().astype(np.uint8)

            submatrix = cls.from_variants_and_haplotypes(
                variant_ids,
                haplotypes,
                n_variants=len(chunk),
                n_haplotypes=n_haplotypes,
                values=haplotype_values)
            matrix.extend(submatrix)

            if convert_multiallelic_to_biallelic:
                n_alleles_per_variant = np.sum(chunk.alt_seq == ",", axis=1) + 1
                matrix.to_biallelic(n_alleles_per_variant)


        return matrix

    @classmethod
    def from_vcf(cls, vcf_file_name) -> 'SparseHaplotypeMatrix':
        return cls.from_vcf2(vcf_file_name)
        # this method can be removed
        vcf = bnp.open(vcf_file_name, buffer_type=bnp.io.delimited_buffers.PhasedVCFMatrixBuffer)
        all_variant_ids = NpList(dtype=np.uint32)
        all_haplotype_ids = NpList(dtype=np.uint16)
        n_haplotypes = None

        offset = 0
        from kage.util import log_memory_usage_now
        matrix = SparseHaplotypeMatrix.empty()

        for i, chunk in enumerate(vcf.read_chunks(min_chunk_size=500000000)):
            logging.info("Chunk %d, variant %s, %s. %d variants processed" % (i, chunk.chromosome[0], chunk.position[0], offset))
            logging.info("N nonzero haplotypes: %d" % len(all_haplotype_ids))
            log_memory_usage_now("Chunk %d" % i)
            genotypes = chunk.genotypes.raw()
            n_haplotypes = genotypes.shape[1] * 2
            # encoding in BioNumPy is 0: "0|0", 1: "0|1", 2: "1|0", 3: "1|1"

            # 0 | 1 or 1 | 1
            variant_ids, individuals = np.where((genotypes == 1) | (genotypes == 3))
            haplotypes = 2 * individuals + 1
            #variant_ids += offset
            #all_variant_ids.extend(variant_ids)
            #all_haplotype_ids.extend(haplotypes)
            submatrix1 = cls.from_variants_and_haplotypes(variant_ids, haplotypes, n_variants=len(chunk), n_haplotypes=n_haplotypes)

            # 1 | 0 or 1 | 1
            variant_ids, individuals = np.where((genotypes == 2) | (genotypes == 3))
            haplotypes = 2 * individuals
            #variant_ids += offset
            #all_variant_ids.extend(variant_ids)
            #all_haplotype_ids.extend(haplotypes)

            offset += len(chunk)
            submatrix2 = cls.from_variants_and_haplotypes(variant_ids, haplotypes, n_variants=len(chunk), n_haplotypes=n_haplotypes)

            matrix.extend(submatrix1+submatrix2)

        logging.info(f"In total {offset} variants and {n_haplotypes} haplotypes")

        return matrix
        return cls.from_variants_and_haplotypes(all_variant_ids.get_nparray(),
                                                all_haplotype_ids.get_nparray(),
                                                n_variants=offset,
                                                n_haplotypes=n_haplotypes)


@dataclass
class GenotypeMatrix:
    # rows are variants, columns are haplotypes
    # 0  = 0/0, 1=0/1 or 1/0, 2=1/1, 4=missing
    matrix: np.ndarray

    @classmethod
    def from_haplotype_matrix(cls, haplotype_matrix):
        n_variants, n_haplotypes = haplotype_matrix.shape
        n_individuals = n_haplotypes // 2
        matrix = np.zeros((n_variants, n_individuals), dtype=np.uint8)
        for i in tqdm.tqdm(range(n_individuals), desc="Making genotype matrix", total=n_individuals, unit="individuals"):
            matrix[:, i] = haplotype_matrix.get_haplotype(i * 2) + haplotype_matrix.get_haplotype(i * 2 + 1)

        return cls(matrix)


def make_sparse_haplotype_matrix_cli(args):
    matrix = SparseHaplotypeMatrix.from_vcf(args.vcf_file_name)
    matrix.to_file(args.out_file_name)

