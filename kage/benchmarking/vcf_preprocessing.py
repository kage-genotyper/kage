import dataclasses
import logging
from isal import igzip
from kage.preprocessing.variants import find_end_in_info_string, get_af_from_info_string_in_vcf_chunk
from tqdm import tqdm

from ..preprocessing.variants import VariantStream, get_padded_variants_from_vcf, find_snps_indels_covered_by_svs

logging.basicConfig(level=logging.INFO)
import bionumpy as bnp
import typing as tp
import numpy as np
from ..io import VcfWithSingleIndividualBuffer, \
    VcfWithInfoBuffer, CustomVCFBuffer


def get_cn0_ref_alt_sequences_from_vcf(variants: bnp.datatypes.VCFEntry,
                                       reference_genome: bnp.genomic_data.GenomicSequence) -> tp.Tuple[
    bnp.EncodedArray, bnp.EncodedArray]:
    """
    Returns reference and alt sequences given variants where all alt-sequences are CN0
    """

    ref = []
    alt = []
    n_wrong = 0

    # process cn0-entries
    for variant in tqdm(variants):
        end = find_end_in_info_string(
            variant.info.to_string())  # end is one based and inclusive in vcf, so now it is 0-based and exlusive
        # print("ENd", end)
        # new_variant_ref_seq = reference_genome.extract_intervals(bnp.Interval([variant.chromosome], [variant.position], [end]))[0].to_string()
        # print("New ref seq: ", new_variant_ref_seq)
        try:
            new_variant_ref_seq = reference_genome[variant.chromosome.to_string()][variant.position:end].to_string()
        except ValueError:
            logging.error(variant)
            raise
        # assert reference_genome[variant.chromosome.to_string()][variant.position].to_string() == variant.ref_seq.to_string()
        if new_variant_ref_seq[0].lower() != variant.ref_seq.to_string().lower():
            logging.error(variant)
            logging.error("New variant ref seq: " + new_variant_ref_seq[0:10] + "...")
            # ignore
            n_wrong += 1

        new_variant_alt_seq = new_variant_ref_seq[0]
        # print("New alt seq", new_variant_alt_seq)

        ref.append(new_variant_ref_seq)
        alt.append(new_variant_alt_seq)

    logging.info(f"{n_wrong} variants had wrong ref sequence ")

    return ref, alt


def _get_sv_mask_and_cn0_mask(variants):
    unknown_sequence = np.any(variants.ref_seq == ">", axis=1) | \
                       np.any(variants.alt_seq == ">", axis=1) | \
                       np.any(variants.ref_seq == "<", axis=1) | \
                       np.any(variants.alt_seq == "<", axis=1) | \
                       np.any(variants.alt_seq == "N", axis=1) | \
                       np.any(variants.alt_seq == "n", axis=1) | \
                       np.any(variants.ref_seq == "N", axis=1) | \
                       np.any(variants.ref_seq == "n", axis=1)
    cn0_entries = bnp.str_equal(variants.alt_seq, "<CN0>")

    to_keep = ~unknown_sequence | cn0_entries

    return to_keep, cn0_entries


def _preprocess_sv_vcf(variants: bnp.datatypes.VCFEntry,
                       reference_genome: bnp.genomic_data.GenomicSequence) -> bnp.datatypes.VCFEntry:
    """
    Preprocesses a sv vcf by removing entries with unknown sequences.
    Keeps only entries with known sequences and <CN0> (replaces those with sequences from reference)
    """

    to_keep, cn0_entries = _get_sv_mask_and_cn0_mask(variants)

    cn0_ref, cn0_alt = get_cn0_ref_alt_sequences_from_vcf(variants[cn0_entries], reference_genome)

    new_cn0_variants = dataclasses.replace(variants[cn0_entries], ref_seq=cn0_ref, alt_seq=cn0_alt)
    variants[cn0_entries] = new_cn0_variants

    return variants[to_keep]


def preprocess_sv_vcf(vcf_file_name, reference_file_name):
    to_keep = []
    cn0_mask = []
    new_ref = []
    new_alt = []

    # reference = bnp.Genome.from_file(reference_file_name).read_sequence()
    logging.info("Reading reference")
    reference = bnp.open(reference_file_name).read()
    reference = {
        r.name.to_string(): r.sequence for r in reference
    }
    logging.info("Done reading")
    # print(reference)

    # first read vcf and find what to keep and new ref/alt sequences
    vcf = bnp.open(vcf_file_name, buffer_type=VcfWithSingleIndividualBuffer)
    for chunk in vcf.read_chunks(min_chunk_size=200000000):
        mask, cn0 = _get_sv_mask_and_cn0_mask(chunk)
        to_keep.append(mask)
        cn0_mask.append(cn0)
        logging.info(f"{np.sum(cn0)} variants with <cn0> alt")
        if np.sum(cn0) == 0:
            continue
        ref, alt = get_cn0_ref_alt_sequences_from_vcf(chunk[cn0], reference)
        new_ref.extend(ref)
        new_alt.extend(alt)
        assert len(ref) == len(alt)
        assert len(ref) == np.sum(cn0)

    to_keep = np.concatenate(to_keep)
    cn0_mask = np.concatenate(cn0_mask)

    # read vcf again, filter, change ref/alt
    cn0_index = 0
    n_wrong = 0
    n_all_genotypes_missing = 0
    with igzip.open(vcf_file_name, "rb") as f:
        i = -1
        for line in f:
            line = line.decode("utf-8").strip()
            if line.startswith("#"):
                print(line)
                continue
            i += 1

            if not to_keep[i]:
                continue

            line = line.split("\t")
            if cn0_mask[i]:
                # replace ref/alt with new sequences
                line[3] = new_ref[cn0_index]
                line[4] = new_alt[cn0_index]
                cn0_index += 1
                # ignore if new sequence contains n
                if "n" in line[3].lower() or "n" in line[4].lower():
                    continue

            if line[3][0].lower() != reference[line[0]][int(line[1]) - 1].to_string().lower():
                logging.error("Reference seq for variant does not match reference sequences. Will be ignored")
                logging.error(line)
                n_wrong += 1
                continue

            # Skip variants where all genotypes are missing
            if all(g == "./." or g == "." for g in line[10:]):
                n_all_genotypes_missing += 1
                continue

            print("\t".join(line))

    logging.info(f"{n_wrong} variants had wrong ref sequence and were ignored ")
    logging.info(f"{n_all_genotypes_missing} variants had all genotypes missing and were ignored ")


def filter_snps_indels_covered_by_svs_cli(args):
    variants = bnp.open(args.vcf).read_chunks(min_chunk_size=200000000)

    remove = []
    for chromosome, variants in bnp.groupby(variants, "chromosome"):
        is_covered = find_snps_indels_covered_by_svs(variants, args.sv_size_limit)
        remove.append(is_covered)
        logging.info(f"Removing {np.sum(is_covered)} variants on chromosome {chromosome}")

    remove = np.concatenate(remove)

    with igzip.open(args.vcf, "rb") as f:
        for line in f:
            line = line.decode("utf-8").strip()
            if line.startswith("#"):
                print(line)
                continue
            if not remove[0]:
                print(line)
            remove = remove[1:]

def find_multiallelic_alleles_with_low_allele_frequency_by_padding_variants(vcf_file_name, reference_fasta, min_allele_frequency: float = 0.05):
    """
    Same as function below, but pads variants in order to find multiallelic overlaps
    """
    variant_stream = VariantStream.from_vcf(vcf_file_name, buffer_type=CustomVCFBuffer)
    variants, vcf_variants, n_alleles_per_original_variant = get_padded_variants_from_vcf(variant_stream,
                                                                                          reference_fasta,
                                                                                          True,
                                                                                          remove_indel_padding=False)
    assert len(variants) == len(vcf_variants), "This is only supported when input is biallelic"

    # todo: This should be done by chromosome
    # need to get info from raw variants so that we know allele frequency
    # can be done chunk by chunk
    afs = []
    raw_variants = bnp.open(vcf_file_name, buffer_type=VcfWithInfoBuffer)
    for chunk in raw_variants:
        logging.info("Getting afs")
        afs.append(get_af_from_info_string_in_vcf_chunk(chunk))
    afs = np.concatenate(afs)
    #variants.info = raw_variants.info  # need info field for filtering
    #filter = find_multiallelic_alleles_with_low_allele_frequency(variants, min_allele_frequency)
    filter = filter_variants_on_same_pos_on_af(afs, min_allele_frequency, variants)
    print_filtered_vcf(filter, vcf_file_name)


def find_multiallelic_alleles_with_low_allele_frequency(biallelic_vcf_entry, min_allele_frequency: float = 0.05, only_deletions=False):
    """
    Returns a boolean mask where True are alleles with frequency lower than min_allele_frequency
    Will only mark variants that are part of multiallelic variants. Will always keep
    at least one biallelic allele (the one with highest allele frequency)
    """
    vcf_entry = biallelic_vcf_entry
    allele_frequencies = get_af_from_info_string_in_vcf_chunk(vcf_entry)
    return filter_variants_on_same_pos_on_af(allele_frequencies, min_allele_frequency, vcf_entry, only_deletions=only_deletions)


def filter_variants_on_same_pos_on_af(allele_frequencies: np.ndarray, min_allele_frequency, vcf_entry, only_deletions=False):
    filter = np.zeros(len(vcf_entry), dtype=bool)
    index = 0
    for pos, variant_group in bnp.groupby(vcf_entry, "position"):
        group_af = allele_frequencies[index:index + len(variant_group)]
        filter_out = group_af < min_allele_frequency
        if only_deletions:
            filter_out = filter_out & (variant_group.ref_seq.shape[1] > 1)
        else:
            # only if not only deletion:
            # always keep best if there are more than 1 variant
            #if len(variant_group) > 1:
            # always keep the best
            filter_out[np.argmax(group_af)] = False
        filter[index:index + len(variant_group)] = filter_out
        index += len(variant_group)
    logging.info(f"Removing {np.sum(filter)} variants")
    return filter


def filter_low_frequency_alleles_on_multiallelic_variants(vcf_file_name, min_allele_frequency: float = 0.05, only_deletions=False):
    """
    Writes to stdout
    """
    chunks = bnp.open(vcf_file_name, buffer_type=VcfWithInfoBuffer).read_chunks()
    all_filters = []
    for chromosome, chromosome_chunk in bnp.groupby(chunks, "chromosome"):
        filter = find_multiallelic_alleles_with_low_allele_frequency(chromosome_chunk, min_allele_frequency, only_deletions=only_deletions)
        logging.info(f"Removing {np.sum(filter)} variants on chromosome {chromosome}")
        all_filters.append(filter)

    all_filters = np.concatenate(all_filters)
    print_filtered_vcf(all_filters, vcf_file_name)


def print_filtered_vcf(all_filters, vcf_file_name):
    i = 0
    with igzip.open(vcf_file_name, "rb") as f:
        for line in f:
            line = line.decode("utf-8")
            if line.startswith("#"):
                print(line.strip())
                continue
            if not all_filters[i]:
                print(line.strip())
            i += 1


def filter_low_frequency_alleles_on_multiallelic_variants_cli(args):
    if args.reference is not None and False:
        logging.info("Will pad variants")
        find_multiallelic_alleles_with_low_allele_frequency_by_padding_variants(args.vcf_file_name, args.reference, args.min_frequency)
    else:
        filter_low_frequency_alleles_on_multiallelic_variants(args.vcf_file_name, args.min_frequency, only_deletions=args.only_deletions)