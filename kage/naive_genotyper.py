"""
A naive genotyper that just sets all genotype likelihoods to 1/3.
Used to have a baseline to compare against when this is used with GLIMPSE
"""
import numpy as np

from .glimpse.glimpse_wrapper import run_glimpse
from .io import write_vcf, SimpleVcfEntry, create_vcf_header_with_sample_name
from .preprocessing.variants import Variants
import bionumpy as bnp


def run_naive_genotyper(population_vcf_file_name: str, out_file_name: str, with_glimpse=False, glimpse_chunks=None):
    variants = SimpleVcfEntry.from_vcf(population_vcf_file_name)
    string_genotypes = bnp.as_encoded_array(["0/0" for _ in range(len(variants))])

    vcf_header = bnp.open(population_vcf_file_name).read_chunk().get_context("header")
    header = create_vcf_header_with_sample_name(vcf_header, "sample",
                                       add_genotype_likelyhoods=True)

    likelihoods = np.log(np.ones((len(variants), 3)) / 3)

    genotype_out_file_name = out_file_name
    if with_glimpse:
        genotype_out_file_name = out_file_name + ".tmp.vcf"

    write_vcf(variants,
              string_genotypes,
              genotype_out_file_name,
              header,
              likelihoods)

    if with_glimpse:
        chromosomes = list(set([variant.chromosome.to_string() for variant in variants]))
        print("Writing to ", out_file_name)
        run_glimpse(population_vcf_file_name, genotype_out_file_name,
                    out_file_name,
                    chromosomes=chromosomes,
                    glimpse_index_dir=glimpse_chunks
                    )

def naive_genotyper_cli(args):
    run_naive_genotyper(args.population_vcf, args.output_vcf, with_glimpse=args.glimpse,
                        glimpse_chunks=args.glimpse_chunks)
