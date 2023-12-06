import logging
from dataclasses import dataclass
import bionumpy as bnp
import numba
import numpy as np
import scipy
import tqdm
from graph_kmer_index.nplist import NpList

from kage.indexing.paths import PathCreator
from kage.preprocessing.variants import VariantStream
from kage.util import log_memory_usage_now


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

    def to_biallelic(self, n_alleles_per_variant, missing_data_encoding=127) -> 'SparseHaplotypeMatrix':
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

        # If missing data at an allele, all alleles in biallelic should also have missing data

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
        new_matrix = (new_matrix == allele_indexes_matrix)
        if self.data.dtype == np.uint8:
            assert np.all(new_matrix) < 256
        elif self.data.dtype == np.uint16:
            assert np.all(new_matrix) < 10000
        new_matrix = new_matrix.astype(self.data.dtype)


        @numba.jit(nopython=True)
        def fill_missing(multiallelic_matrix, biallelic_matrix, n_alleles_per_variant, missing_data_encoding):
            # all alleles that were missing at multiallelic should be missing at all alleles in biallelic
            for haplotype in range(biallelic_matrix.shape[1]):
                i = 0
                for variant in range(len(n_alleles_per_variant)):
                    if multiallelic_matrix[variant, haplotype] == missing_data_encoding:
                        for allele in range(n_alleles_per_variant[variant]-1):
                            biallelic_matrix[i+allele, haplotype] = missing_data_encoding
                    i += n_alleles_per_variant[variant]-1

        fill_missing(matrix, new_matrix, n_alleles_per_variant, missing_data_encoding)

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
        matrix = np.asarray(matrix)
        assert np.all(matrix >= 0), "Values below 0"
        return cls(scipy.sparse.csc_matrix(matrix))

    def to_matrix(self):
        return self.data.toarray()

    @classmethod
    def from_variants_and_haplotypes(cls, variant_ids, haplotype_ids, n_variants, n_haplotypes, values=None, dtype=np.uint8):
        """
        variant_ids: np.array of variant ids
        haplotype_ids: np.array of haplotype ids (which haplotypes have the variant allele)
        values: If None, will be filled with ones
        """
        if values is None:
            values = np.ones(len(variant_ids), dtype=dtype)
        else:
            values = values.astype(dtype)
            assert len(values) == len(variant_ids)
        data = scipy.sparse.csc_matrix((values, (variant_ids, haplotype_ids)),
                                       shape=(n_variants, n_haplotypes),
                                       dtype=dtype)
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
    def from_vcf2(cls, vcf_file_name, convert_multiallelic_to_biallelic=False, dtype=np.uint8) -> 'SparseHaplotypeMatrix':
        # Uses haplotypeencoding
        if isinstance(vcf_file_name, str):
            variants = bnp.open(vcf_file_name, buffer_type=bnp.io.vcf_buffers.PhasedHaplotypeVCFMatrixBuffer).read_chunks(min_chunk_size=500000000)
        else:
            assert isinstance(vcf_file_name, VariantStream)
            variants = vcf_file_name.read_chunks()

        matrix = SparseHaplotypeMatrix.empty()

        n_alleles_per_variant = []

        for i, chunk in enumerate(variants):
            genotypes = chunk.genotypes.raw()
            n_haplotypes = genotypes.shape[1]
            variant_ids, haplotypes = np.where(genotypes > 0)
            haplotype_values = genotypes[variant_ids, haplotypes].ravel().astype(dtype)

            submatrix = cls.from_variants_and_haplotypes(
                variant_ids,
                haplotypes,
                n_variants=len(chunk),
                n_haplotypes=n_haplotypes,
                values=haplotype_values)
            matrix.extend(submatrix)
            #logging.info(f"{len(chunk)*len(genotypes)} values, {np.sum(haplotype_values != 0)} nonzero. Size sparse matrix: {submatrix.data.data.nbytes / 1000000000} GB")

            if convert_multiallelic_to_biallelic:
                n_alleles_per_variant = np.sum(chunk.alt_seq == ",", axis=1) + 1
                matrix.to_biallelic(n_alleles_per_variant)

        return matrix

    @classmethod
    def from_vcf(cls, vcf_file_name, dtype=np.uint8) -> 'SparseHaplotypeMatrix':
        return cls.from_vcf2(vcf_file_name, dtype=dtype)


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

