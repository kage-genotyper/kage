import resource
import logging
import bionumpy as bnp
import numpy as np
from shared_memory_wrapper.util import interval_chunks
from collections import namedtuple


def get_memory_usage():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000

def log_memory_usage_now(logplace=""):
    memory = get_memory_usage()
    logging.info("Memory usage (%s): %.4f GB" % (logplace, memory))



def vcf_pl_and_gl_header_lines():
    return ['##FILTER=<ID=LowQUAL,Description="Quality is low">',
     '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="PHRED-scaled genotype likelihoods.">',
     '##FORMAT=<ID=GL,Number=G,Type=Float,Description="Genotype likelihoods.">',
     '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality: phred scaled probability that the genotype is wrong.">'
    ]


def convert_string_genotypes_to_numeric_array(genotypes):
    numeric_genotypes = ["0/0", "0/0", "1/1", "0/1"]
    numpy_genotypes = np.array([numeric_genotypes[g] for g in genotypes], dtype="|S3")
    return numpy_genotypes


def _write_genotype_debug_data(genotypes, numpy_genotypes, out_name, variant_to_nodes, probs, count_probs):

    np.save(out_name + ".genotypes", genotypes)
    np.save(out_name + ".probs", probs)
    np.save(out_name + ".count_probs", count_probs)
    return

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


def zip_sequences(a: bnp.EncodedRaggedArray, b: bnp.EncodedRaggedArray, encoding = bnp.DNAEncoding):
    """Utility function for merging encoded ragged arrays ("zipping" rows).
    a and b should either be equal size or a can be 1 element longer than b (then first and last element will be from a)"""
    assert len(a) == len(b)+1 or len(a) == len(b)

    row_lengths = np.zeros(len(a)+len(b))
    row_lengths[0::2] = a.shape[1]
    row_lengths[1::2] = b.shape[1]

    # empty placeholder to fill

    new = bnp.EncodedRaggedArray(
        bnp.EncodedArray(np.zeros(int(np.sum(row_lengths)), dtype=np.uint8), encoding),
        row_lengths)
    new[0::2] = a
    if len(a) == len(b) + 1:
        new[1:-1:2] = b
    else:
        new[1::2] = b
    return new


def stream_ragged_array(a, chunk_size=100000):
    chunks = interval_chunks(0, len(a), 1+len(a) // chunk_size)
    for chunk in chunks:
        yield a[chunk[0]:chunk[1]]


def n_unique_values_per_column(matrix: np.ndarray):
    """
    Finds number of unique values per columnn by sorting. Fast when n rows is small
    """
    sorted = np.sort(matrix, axis=0)
    unique = np.sum(np.diff(sorted, axis=0) != 0, axis=0) + 1
    return unique

def make_args_for_genotype_command(index_file_name, reads_file_name, out_file="test_genotypes.vcf", average_coverage=15, kmer_size=31):
    Args = namedtuple("args", ["index_bundle",
                               "reads",
                               "out_file_name",
                               "kmer_size",
                               "average_coverage",
                               "debug",
                               "n_threads",
                               "use_naive_priors",
                               "ignore_helper_model",
                               "ignore_helper_variants",
                               "min_genotype_quality",
                               "sample_name_output",
                               "ignore_homo_ref",
                               "do_not_write_genotype_likelihoods",
                               "gpu",
                               "counts",
                               "write_debug_data",
                               "glimpse",
                               "only_impute_svs"])
    args = Args(index_file_name, reads_file_name, "test_genotypes.vcf", kmer_size, average_coverage, True, 4,
                False, False, False, 0, "sample", False, False, False, None, True, None, False)
    return args
