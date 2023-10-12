import dataclasses
import logging
from isal import igzip
from tqdm import tqdm
import npstructures as nps
logging.basicConfig(level=logging.INFO)
import bionumpy as bnp
import typing as tp
import numpy as np
import re


def find_end_in_info_string(info_string: str) -> int:
    fields = info_string.split(";")
    fields = [f for f in fields if f.startswith("END=")]
    assert len(fields) == 1
    value = fields[0].split("=")[1]
    return int(value)


def get_cn0_ref_alt_sequences_from_vcf(variants: bnp.datatypes.VCFEntry, reference_genome: bnp.genomic_data.GenomicSequence) -> tp.Tuple[bnp.EncodedArray, bnp.EncodedArray]:
    """
    Returns reference and alt sequences given variants where all alt-sequences are CN0
    """

    ref = []
    alt = []
    n_wrong = 0

    # process cn0-entries
    for variant in tqdm(variants):
        end = find_end_in_info_string(variant.info.to_string())  # end is one based and inclusive in vcf, so now it is 0-based and exlusive
        #print("ENd", end)
        #new_variant_ref_seq = reference_genome.extract_intervals(bnp.Interval([variant.chromosome], [variant.position], [end]))[0].to_string()
        #print("New ref seq: ", new_variant_ref_seq)
        try:
            new_variant_ref_seq = reference_genome[variant.chromosome.to_string()][variant.position:end].to_string()
        except ValueError:
            logging.error(variant)
            raise
        #assert reference_genome[variant.chromosome.to_string()][variant.position].to_string() == variant.ref_seq.to_string()
        if new_variant_ref_seq[0].lower() != variant.ref_seq.to_string().lower():
            logging.error(variant)
            logging.error("New variant ref seq: " + new_variant_ref_seq[0:10] + "...")
            # ignore
            n_wrong += 1

        new_variant_alt_seq = new_variant_ref_seq[0]
        #print("New alt seq", new_variant_alt_seq)

        ref.append(new_variant_ref_seq)
        alt.append(new_variant_alt_seq)

    logging.info(f"{n_wrong} variants had wrong ref sequence ")

    return ref, alt


def _get_sv_mask_and_cn0_mask(variants):
    unknown_sequence = np.any(variants.ref_seq == ">", axis=1) | \
                       np.any(variants.alt_seq == ">", axis=1) | \
                       np.any(variants.ref_seq == "<", axis=1) | \
                       np.any(variants.alt_seq == "<", axis=1)

    cn0_entries = bnp.str_equal(variants.alt_seq, "<CN0>")

    to_keep = ~unknown_sequence | cn0_entries

    return to_keep, cn0_entries


def _preprocess_sv_vcf(variants: bnp.datatypes.VCFEntry, reference_genome: bnp.genomic_data.GenomicSequence) ->  bnp.datatypes.VCFEntry:
    """
    Preprocesses a sv vcf by removing entries with unknown sequences.
    Keeps only entries with known sequences and <CN0> (replaces those with sequences from reference)
    """

    unknown_sequence = np.any(variants.ref_seq == ">", axis=1) | \
                       np.any(variants.alt_seq == ">", axis=1) | \
                       np.any(variants.ref_seq == "<", axis=1) | \
                       np.any(variants.alt_seq == "<", axis=1)

    cn0_entries = bnp.str_equal(variants.alt_seq, "<CN0>")

    to_keep = ~unknown_sequence | cn0_entries

    cn0_ref, cn0_alt = get_cn0_ref_alt_sequences_from_vcf(variants[cn0_entries], reference_genome)

    new_cn0_variants = dataclasses.replace(variants[cn0_entries], ref_seq=cn0_ref, alt_seq=cn0_alt)
    variants[cn0_entries] = new_cn0_variants


    return variants[to_keep]


def preprocess_sv_vcf(vcf_file_name, reference_file_name):
    to_keep = []
    cn0_mask = []
    new_ref = []
    new_alt = []

    #reference = bnp.Genome.from_file(reference_file_name).read_sequence()
    logging.info("Reading reference")
    reference = bnp.open(reference_file_name).read()
    reference = {
        r.name.to_string(): r.sequence for r in reference
    }
    logging.info("Done reading")
    #print(reference)

    # first read vcf and find what to keep and new ref/alt sequences
    vcf = bnp.open(vcf_file_name)
    for chunk in vcf.read_chunks(min_chunk_size=200000000):
        mask, cn0 = _get_sv_mask_and_cn0_mask(chunk)
        to_keep.append(mask)
        cn0_mask.append(cn0)
        logging.info(f"{np.sum(cn0)} variants with <cn0> alt")
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

            if line[3][0].lower() != reference[line[0]][int(line[1]) - 1].to_string().lower():
                logging.error("Reference seq for variant does not match reference sequences. Will be ignored")
                logging.error(line)
                n_wrong += 1
                continue

            print("\t".join(line))

    logging.info(f"{n_wrong} variants had wrong ref sequence and were ignored ")


def find_snps_indels_covered_by_svs(variants: bnp.datatypes.VCFEntry, sv_size_limit: int = 50) -> np.ndarray:
    """
    Returns a boolean mask where True are SNPs/indels that are covered by a SV.
    Assumes all variants are on the same chromosome.
    """
    assert variants.chromosome[0].to_string() == variants.chromosome[-1].to_string()
    is_snp_indel = (variants.ref_seq.shape[1] <= sv_size_limit) & (variants.alt_seq.shape[1] <= sv_size_limit)
    is_sv = ~is_snp_indel

    is_any_indel = (variants.ref_seq.shape[1] > 1) | (variants.alt_seq.shape[1] > 1)
    starts = variants.position
    starts[is_any_indel] += 1  # indels are padded with one base
    ends = starts + variants.ref_seq.shape[1] - 1

    sv_position_mask = np.zeros(np.max(ends)+1, dtype=bool)
    indexes_of_covered_by_sv = nps.ragged_slice(np.arange(len(sv_position_mask)), starts[is_sv], ends[is_sv]).ravel()
    sv_position_mask[indexes_of_covered_by_sv] = True

    is_covered = (sv_position_mask[starts] | sv_position_mask[ends]) & is_snp_indel

    return is_covered


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

