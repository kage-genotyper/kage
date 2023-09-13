import resource
import logging
import bionumpy as bnp
import numpy as np
from shared_memory_wrapper.util import interval_chunks


def get_memory_usage():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000

def log_memory_usage_now(logplace=""):
    memory = get_memory_usage()
    logging.info("Memory usage (%s): %.4f GB" % (logplace, memory))



def vcf_pl_and_gl_header_lines():
    return ['##FILTER=<ID=LowQUAL,Description="Quality is low">',
     '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="PHRED-scaled genotype likelihoods.">',
     '##FORMAT=<ID=GL,Number=G,Type=Float,Description="Genotype likelihoods.">'
     ]


def convert_string_genotypes_to_numeric_array(genotypes):
    numeric_genotypes = ["0/0", "0/0", "1/1", "0/1"]
    numpy_genotypes = np.array([numeric_genotypes[g] for g in genotypes], dtype="|S3")
    return numpy_genotypes


def _write_genotype_debug_data(genotypes, numpy_genotypes, out_name, variant_to_nodes, probs, count_probs):

    np.save(out_name + ".probs", probs)
    np.save(out_name + ".count_probs", count_probs)

    # Make arrays with haplotypes
    haplotype_array1 = np.zeros(len(numpy_genotypes), dtype=np.uint8)
    haplotype_array1[np.where((genotypes == 2) | (genotypes == 3))[0]] = 1
    haplotype_array2 = np.zeros(len(numpy_genotypes), dtype=np.uint8)
    haplotype_array2[np.where(genotypes == 2)[0]] = 1
    np.save(out_name + ".haplotype1", haplotype_array1)
    np.save(out_name + ".haplotype2", haplotype_array2)
    # also store variant nodes from the two haplotypes
    variant_nodes_haplotype1 = variant_to_nodes.var_nodes[np.nonzero(haplotype_array1)]
    variant_nodes_haplotype2 = variant_to_nodes.var_nodes[np.nonzero(haplotype_array2)]
    np.save(out_name + ".haplotype1_nodes", variant_nodes_haplotype1)
    np.save(out_name + ".haplotype2_nodes", variant_nodes_haplotype2)


def zip_sequences(a: bnp.EncodedRaggedArray, b: bnp.EncodedRaggedArray):
    """Utility function for merging encoded ragged arrays ("zipping" rows)"""
    assert len(a) == len(b)+1

    row_lengths = np.zeros(len(a)+len(b))
    row_lengths[0::2] = a.shape[1]
    row_lengths[1::2] = b.shape[1]

    # empty placeholder to fill

    new = bnp.EncodedRaggedArray(
        bnp.EncodedArray(np.zeros(int(np.sum(row_lengths)), dtype=np.uint8), bnp.DNAEncoding),
        row_lengths)
    new[0::2] = a
    new[1:-1:2] = b
    return new


def stream_ragged_array(a, chunk_size=100000):
    chunks = interval_chunks(0, len(a), 1+len(a) // chunk_size)
    for chunk in chunks:
        yield a[chunk[0]:chunk[1]]
