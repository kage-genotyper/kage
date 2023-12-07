"""
A naive genotyper that just sets all genotype likelihoods to 1/3.
Used to have a baseline to compare against when this is used with GLIMPSE
"""
import numpy as np

from .io import write_vcf, SimpleVcfEntry, create_vcf_header_with_sample_name
from .preprocessing.variants import Variants
import bionumpy as bnp


def run_naive_genotyper(population_vcf_file_name: str, out_file_name: str):
    variants = SimpleVcfEntry.from_vcf(population_vcf_file_name)
    string_genotypes = bnp.as_encoded_array(["0/0" for _ in range(len(variants))])

    vcf_header = bnp.open(population_vcf_file_name).read_chunk().get_context("header"),
    header = create_vcf_header_with_sample_name(vcf_header, "sampl",
                                       add_genotype_likelyhoods=True),

    likelihoods = np.log(np.ones((len(variants), 3)) / 3)
    write_vcf(variants,
              string_genotypes,
              out_file_name,
              header,
              likelihoods)


def naive_genotyper_cli(args):
    run_naive_genotyper(args.population_vcf, args.output_vcf)
