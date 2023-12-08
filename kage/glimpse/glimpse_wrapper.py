"""
A wrapper script around GLIMPSE, enabling running GLIMPSE through Python
"""
import glob
import logging
import multiprocessing
import os
import subprocess
import urllib
from collections import defaultdict
from dataclasses import dataclass
import random
from tempfile import TemporaryFile, NamedTemporaryFile
from typing import List
import bionumpy as bnp
import numpy as np

GLIMPSE_BINARIES = {
    "GLIMPSE_chunk_static": "https://github.com/odelaneau/GLIMPSE/releases/download/v1.1.1/GLIMPSE_chunk_static",
    "GLIMPSE_phase_static": "https://github.com/odelaneau/GLIMPSE/releases/download/v1.1.1/GLIMPSE_phase_static",
    "GLIMPSE_ligate_static": "https://github.com/odelaneau/GLIMPSE/releases/download/v1.1.1/GLIMPSE_ligate_static",
}


def get_vcf_chromosomes(vcf_file_name):
    chunk = bnp.open(vcf_file_name).read_chunk()
    header = chunk.get_context("header")
    chromosome_header_lines = [line for line in header.split("\n") if line.lower().startswith("##contig")]
    assert len(chromosome_header_lines) > 0, "No chromosome header lines found in VCF file. VCF file needs to contain chromosomes in the header."
    chromosomes = [line.split("ID=")[1].split(",")[0] for line in chromosome_header_lines]
    logging.info("Assuming chromosomes are %s based on VCF header" % chromosomes)
    return chromosomes


def download_glimpse_binaries_if_not_exist():
    for binary_name, binary_url in GLIMPSE_BINARIES.items():
        if not os.path.isdir("glimpse_binaries"):
            os.mkdir("glimpse_binaries")
        file_name = "glimpse_binaries/" + binary_name
        if not os.path.exists(file_name):
            logging.info("Downloading %s" % binary_name)
            urllib.request.urlretrieve(binary_url, file_name)
            os.chmod(file_name, 0o755)
        else:
            logging.info("Binary %s already exists, not downloading" % binary_name)


def make_glimps_chunks_for_chromosome(vcf_file_name: str, chromosome, out_dir: str):
    subprocess.run(["glimpse_binaries/GLIMPSE_chunk_static",
                    "--input", vcf_file_name,
                    "--region", chromosome,
                    #"--window-size", "20000",
                    #"--buffer-size", "20000",
                    "--window-count", "1000",
                    "--buffer-count", "100",
                    "--output", out_dir + "/glimpse_chunks." + chromosome + ".txt"
                    ])


def setup_glimpse(out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    download_glimpse_binaries_if_not_exist()


def make_glimpse_chunks(vcf_file_name: str, out_dir: str, n_threads: int = 1, chromosomes = None):
    setup_glimpse(out_dir)
    chromosomes = get_vcf_chromosomes(vcf_file_name) if chromosomes is None else chromosomes

    pool = multiprocessing.Pool(n_threads)
    for chromosome in chromosomes:
        logging.info("Running for chromosome %s" % chromosome)
        pool.apply_async(make_glimps_chunks_for_chromosome, (vcf_file_name, chromosome, out_dir))

    pool.close()
    pool.join()


def bgzip_vcf(vcf_file_name: str):
    subprocess.run(["bgzip", "-f", vcf_file_name])
    tabix_vcf(vcf_file_name + ".gz")


def remove_glimpse_results(out_dir: str):
    """
    Removes old glimpse results. Important to not mix old vcf files with new ones
    """
    for file in glob.glob(out_dir + "/GLIMPSE*"):
        logging.info("Removing old file %s" % file)
        os.remove(file)


@dataclass
class GlimpseParams:
    id: str
    chromosome: str
    input_region: str
    output_region: str


def run_glimpse_on_chunk(genotyped_vcf: str, population_vcf: str, out_dir: str,
                         genetic_map: str, params: GlimpseParams):
    out_file_name = os.sep.join([out_dir, "GLIMPSE-" + params.chromosome + "." + params.id + ".bcf"])
    params = [
        "glimpse_binaries/GLIMPSE_phase_static",
        "--input-GL",
        "--input", genotyped_vcf,
        "--reference", population_vcf,
        #"--map", genetic_map,
        "--input-region", params.input_region,
        "--output-region", params.output_region,
        "--output", out_file_name
    ]
    if genetic_map != "":
        params += ["--map", genetic_map]
    subprocess.run(params)
    subprocess.run([
        "bcftools", "index", "-f", out_file_name
    ])
    return out_file_name


def run_glimpse(population_vcf: str, genotyped_vcf: str, out_file: str, genetic_map: str = "",
                n_threads: int = 1, chromosomes = None, glimpse_index_dir = None):
    out_path = os.path.dirname(out_file)
    if out_path == "":
        out_path = "./"
    setup_glimpse(out_path)

    if glimpse_index_dir is None:
        logging.info("No GLIMPSE directory provided. Will create chunks (index)")
        make_glimpse_chunks(population_vcf, out_path, n_threads=n_threads, chromosomes=chromosomes)
        glimpse_index_dir = out_path
        chromosomes = get_vcf_chromosomes(population_vcf) if chromosomes is None else chromosomes

    assert chromosomes is not None

    if not genotyped_vcf.endswith(".gz"):
        logging.info("Bgzipping genotyped VCF")
        bgzip_vcf(genotyped_vcf)
        genotyped_vcf += ".gz"

    remove_glimpse_results(out_path)

    pool = multiprocessing.Pool(n_threads)
    created_files = defaultdict(list)
    for chromosome in chromosomes:
        arguments = []
        for line in open(glimpse_index_dir + "/glimpse_chunks." + chromosome + ".txt"):
            params = line.split()
            params = GlimpseParams(*params[0:4])
            arguments.append((genotyped_vcf, population_vcf, out_path, genetic_map, params))

        for result in pool.starmap(run_glimpse_on_chunk, arguments):
            created_files[chromosome].append(result)

            #pool.apply_async(run_glimpse_on_chunk, (genotyped_vcf, population_vcf, out_path, genetic_map, params),
            #                 callback=lambda file: created_files[chromosome].append(file))
            #logging.info("Running on %s"  % params)
            #file = run_glimpse_on_chunk(genotyped_vcf, population_vcf, out_path, genetic_map, params)
            #created_files[chromosome].append(file)

    pool.close()
    pool.join()

    # ligate files on same chromosome
    joint_chromosome_files = []
    for chromosome, files in created_files.items():
        result = run_glimpse_ligate(chromosome, files, out_path)
        joint_chromosome_files.append(result)

    # Concatenate all chromosome files in the end
    bcftools_concat(joint_chromosome_files, out_file)


def tabix_vcf(vcf_file_name: str):
    subprocess.run(["tabix", "-p", "vcf", "-f", vcf_file_name])


def bcftools_concat(vcf_files: List[str], out_file: str):
    subprocess.run(["bcftools", "concat", "-o", out_file] + vcf_files)


def run_glimpse_ligate(chromosome: str, chromosome_bcf_files: List[str], out_dir: str):
    """
    Glimpse needs these bcf-files written to a file
    """
    logging.info("Making tmp file with names for chromosome %s: %s" % (chromosome, chromosome_bcf_files))
    #random_name = str(random.randint(0, 1000000000)) + ".txt"
    with NamedTemporaryFile(mode="w", delete=False) as f:
    #with open(random_name, "w") as f:
        for file in chromosome_bcf_files:
            f.write(file + "\n")

    out_file = out_dir + "/GLIMPSE.ligated." + chromosome + ".vcf.gz"

    subprocess.run([
        "glimpse_binaries/GLIMPSE_ligate_static",
        "--input", f.name,
        "--output", out_file
    ])

    tabix_vcf(out_file)
    return out_file


def run_glimpse_cli(args):
    run_glimpse(args.population_vcf, args.genotyped_vcf, args.output_vcf, "", args.n_threads, args.chromosomes.split(","))


def run_glimpse_index_cli(args):
    variants = bnp.open(args.population_vcf).read()
    chromosomes = list(np.unique(variants.chromosome.raw()).astype(str))
    make_glimpse_chunks(args.population_vcf, args.output_dir, args.n_threads, chromosomes)

