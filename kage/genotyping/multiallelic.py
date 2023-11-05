import logging

import numpy as np
from kage.indexing.main import MultiAllelicMap


def postprocess_multiallelic_calls(genotypes: np.ndarray, multiallelic_map: MultiAllelicMap, probs: np.ndarray):
    """
    Postprocesses genotype calls (biallelic) by making them compatible with
    multi-allelic variants.
    Idea is that variants that are part of the same multi-allelic variant
    need to be compatible (i.e. not conflicting genotypes)
    Uses the genotype probabilities (log-scale) to find the most likely genotypes
    within multiallelic variants

    Genotypes are numeric:
     0 or 1: 0/0, 2: 1/1, 3: 0/1

     Probs is matrix of shape n_variants x 3 where each
     column corresponds to log e prob of genotypes
     (0/0, 0/1, 1/1)


    """
    assert len(genotypes) == len(multiallelic_map.ravel()), len(multiallelic_map.ravel())
    original_genotypes = genotypes.copy()

    # nan-probs will cause problem
    n_nan = np.sum(np.isnan(probs))
    logging.info(f"{n_nan} probs are nan. Setting them to 1/3 before postprocesing genotypes")
    probs[np.isnan(probs)] = np.log(1/3)

    biallelic_id = 0
    n_changed = 0
    for multiallelic in multiallelic_map:
        if len(multiallelic) == 1:
            # no multiallelic, do nothing
            pass
        else:
            variant_probs = probs[biallelic_id:biallelic_id + len(multiallelic)]
            variant_genotypes = genotypes[biallelic_id:biallelic_id + len(multiallelic)]
            # find most likely non-0/0 genotype among all variants
            # if this variant is 0/0, set all genotypes to 0/0
            # if this is 1/1, set all others to 0/0
            # if this is 0/1, check if any others is 0/1 and keep those two as 0/1 and set rest to 0/0

            most_likely_non_homo_ref = np.argmax(np.max(variant_probs, axis=1), axis=0)
            g = variant_genotypes[most_likely_non_homo_ref]

            if g == 0 or g == 1:
                genotypes[biallelic_id:biallelic_id+len(multiallelic)] = 1
                n_changed += 1
            elif g == 2 or g == 3:  # 1/1
                genotypes[biallelic_id:biallelic_id+len(multiallelic)] = 1
                genotypes[biallelic_id+most_likely_non_homo_ref] = g
                n_changed += 1
            else:
                assert False
            #elif g == 1:
            # check if any other variant is 1/1 or 0/1
            # keep that other variant as 0/1 and set all others to 0/0
            #    sorting = np.argsort(np.max(variant_probs, axis=1), axis=0)
            #    other = sorting[1]
            #    if genotypes[]
        biallelic_id += len(multiallelic)

    logging.info(f"Number of genotypes that were changed in postprocessing: {np.sum(genotypes != original_genotypes)}")
    return genotypes, probs
