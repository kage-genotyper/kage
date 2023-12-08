from kage.glimpse.glimpse_wrapper import download_glimpse_binaries_if_not_exist, make_glimpse_chunks, run_glimpse, \
    remove_glimpse_results
import logging
logging.basicConfig(level=logging.DEBUG)


def test():
    download_glimpse_binaries_if_not_exist()


def test_make_glimpse_chunks():
    vcf = "test_data_sacCer3/filtered_population.vcf.gz"
    out_dir = "test_data_sacCer3/glimpse_chunks"
    make_glimpse_chunks(vcf, out_dir, 2)


def test_run_full_glimpse():
    population = "test_data_sacCer3/filtered_population.vcf.gz"
    ref = "test_data_sacCer3/reference.fa"
    genotyped_vcf = "test_data_sacCer3/test_genotypes.vcf.gz"
    run_glimpse(population, genotyped_vcf, "glimpse_test_output/genotypes.vcf.gz", "", n_threads=2)


if __name__ == "__main__":
    remove_glimpse_results("glimpse_test_output/")