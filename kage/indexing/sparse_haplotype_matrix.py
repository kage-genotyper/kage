import logging
from dataclasses import dataclass
import bionumpy as bnp
import numpy as np
import scipy

@dataclass
class SparseHaplotypeMatrix:
    data: scipy.sparse.csc_matrix


    @classmethod
    def from_variants_and_haplotypes(cls, variant_ids, haplotype_ids, n_variants, n_haplotypes):
        """
        variant_ids: np.array of variant ids
        haplotype_ids: np.array of haplotype ids (which haplotypes have the variant allele)
        """
        print(variant_ids)
        print(haplotype_ids)
        data = scipy.sparse.csc_matrix((np.ones(len(variant_ids), dtype=np.uint8), (variant_ids, haplotype_ids)),
                                       shape=(n_variants, n_haplotypes))
        return cls(data)

    def get_haplotype(self, haplotype_id):
        return self.data.getcol(haplotype_id).toarray().flatten()

    @property
    def shape(self):
        return self.data.shape

    def to_file(self, file_name):
        scipy.sparse.save_npz(file_name, self.data)

    @classmethod
    def from_file(cls, file_name):
        data = scipy.sparse.load_npz(file_name)
        return cls(data)


    @classmethod
    def from_vcf(cls, vcf_file_name):
        vcf = bnp.open(vcf_file_name, buffer_type=bnp.io.delimited_buffers.PhasedVCFMatrixBuffer)
        all_variant_ids = []
        all_haplotype_ids = []
        n_haplotypes = None

        offset = 0
        for i, chunk in enumerate(vcf.read_chunks()):
            logging.info(f"Processed {offset} variants")
            genotypes = chunk.genotypes.raw()
            n_haplotypes = genotypes.shape[1] * 2
            # encoding is 0: "0|0", 1: "0|1", 2: "1|0", 3: "1|1"

            # 0 | 1 or 1 | 1
            variant_ids, individuals = np.where((genotypes == 1) | (genotypes == 3))
            haplotypes = 2 * individuals + 1
            variant_ids += offset
            all_variant_ids.append(variant_ids)
            all_haplotype_ids.append(haplotypes)

            # 1 | 0 or 1 | 1
            variant_ids, individuals = np.where((genotypes == 2) | (genotypes == 3))
            haplotypes = 2 * individuals
            variant_ids += offset
            all_variant_ids.append(variant_ids)
            all_haplotype_ids.append(haplotypes)

            offset += len(chunk)


        return cls.from_variants_and_haplotypes(np.concatenate(all_variant_ids),
                                                np.concatenate(all_haplotype_ids),
                                                n_variants=offset,
                                                n_haplotypes=n_haplotypes)

def make_sparse_haplotype_matrix_cli(args):
    matrix = SparseHaplotypeMatrix.from_vcf(args.vcf_file_name)
    matrix.to_file(args.out_file_name)

