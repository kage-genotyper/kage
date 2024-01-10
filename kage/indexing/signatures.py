import logging
import time
from dataclasses import dataclass
import bionumpy as bnp
import ray
from graph_kmer_index import KmerIndex, kmer_hash_to_sequence
import npstructures as nps
import numpy as np
import tqdm
from graph_kmer_index import FlatKmers, CollisionFreeKmerIndex
from shared_memory_wrapper.util import interval_chunks

from .kmer_scoring import Scorer, FastApproxCounter
from .paths import Paths, PathSequences
from ..preprocessing.variants import Variants, VariantAlleleToNodeMap
from typing import List, Union
import awkward as ak
from graph_kmer_index import FlatKmers
import numba

from ..util import get_memory_usage, log_memory_usage_now


@dataclass
class Signatures:
    ref: nps.RaggedArray
    alt: nps.RaggedArray

    def get_overlap(self):
        return np.sum(np.in1d(self.ref.ravel(), self.alt.ravel())) / len(self.ref.ravel())

    def get_as_flat_kmers(self):
        signatures = self
        logging.info("Signature overlap: %.5f percent" % (signatures.get_overlap() * 100))

        all_kmers = []
        all_node_ids = []

        for variant_id, (ref_kmers, variant_kmers) in enumerate(zip(signatures.ref, signatures.alt)):
            node_id_ref = variant_id * 2
            node_id_alt = variant_id * 2 + 1
            all_kmers.append(ref_kmers)
            all_node_ids.append(np.full(len(ref_kmers), node_id_ref, dtype=np.uint32))
            all_kmers.append(variant_kmers)
            all_node_ids.append(np.full(len(variant_kmers), node_id_alt, dtype=np.uint32))

        from graph_kmer_index import FlatKmers
        return FlatKmers(np.concatenate(all_kmers), np.concatenate(all_node_ids))

    def get_as_kmer_index(self, k, modulo=103, include_reverse_complements=True):
        flat = self.get_as_flat_kmers()
        if include_reverse_complements:
            rev_comp_flat = flat.get_reverse_complement_flat_kmers(k)
            flat = FlatKmers.from_multiple_flat_kmers([flat, rev_comp_flat])

        assert modulo > len(flat._hashes), "Modulo is too small for number of kmers. Increase"
        index = KmerIndex.from_flat_kmers(flat, skip_frequencies=True, modulo=modulo)
        index.convert_to_int32()
        return index


@dataclass
class MultiAllelicSignatures:
    # there can be multiple signatures for one allele at a variant
    signatures: ak.Array  # n_variants x n_alleles x signatures

    def describe(self, k):
        from graph_kmer_index import kmer_hash_to_sequence
        for variant_id, variant in enumerate(self.signatures):
            print("Variant ", variant_id)
            for allele_id, allele in enumerate(variant):
                print("  Allele", allele_id, " ".join(kmer_hash_to_sequence(kmer, k) for kmer in allele))

    def to_list_of_sequences(self, k):
        # for testing purposes only
        from graph_kmer_index import kmer_hash_to_sequence
        out = []
        for variant_id, variant in enumerate(self.signatures):
            variant_kmers = []
            for allele_id, allele in enumerate(variant):
                allele_kmers = [kmer_hash_to_sequence(kmer, k) for kmer in allele]
                variant_kmers.append(allele_kmers)
            out.append(variant_kmers)
        return out

    def to_biallelic_list_of_sequences(self, k):
        # for testing purposes only
        from graph_kmer_index import kmer_hash_to_sequence
        out = []
        for variant_id, variant in enumerate(self.signatures):
            ref_kmers = [kmer_hash_to_sequence(kmer, k) for kmer in variant[0]]
            for allele_id, allele in enumerate(variant[1:]):
                allele_kmers = [kmer_hash_to_sequence(kmer, k) for kmer in allele]
                out.append((ref_kmers, allele_kmers))
        return out

    @classmethod
    def from_multiple(cls, signatures: List['MultiAllelicSignatures']):
        """Merges by concatenating"""
        signatures = [s.signatures for s in signatures]
        return cls(np.concatenate(signatures))

    @classmethod
    def from_list(cls, l) -> 'MultiAllelicSignatures':
        a = ak.Array(l)
        return cls(a)

    def to_list(self):
        return ak.to_list(self.signatures)

    def filter_nonunique_on_alleles(self, also_remove=-1):
        # keeps only unique kmers per allele
        signatures = self.signatures
        signatures = np.sort(signatures, axis=2)
        mask = np.ones_like(signatures, dtype=bool)
        flat_mask = ak.to_numpy(ak.ravel(mask))
        n_variants = len(signatures)

        # numba-function for creating mask of non-unique kmers
        @numba.jit(nopython=True)
        def create_mask(flat_mask, signatures):
            i = 0
            for variant in range(n_variants):
                for allele in range(len(signatures[variant])):
                    prev_kmer = np.uint64(0)
                    j = 0
                    for kmer in signatures[variant][allele]:
                        if (kmer == prev_kmer and j > 0) or (also_remove >= 0 and kmer == also_remove):
                            flat_mask[i] = False
                        j += 1
                        prev_kmer = kmer  # signatures[variant, allele, kmer]
                        i += 1

        create_mask(flat_mask, signatures)

        #logging.info("Removed %d non-unique kmers" % np.sum(~mask))
        # use mask to create a new data structure
        """
        keep = ak.ravel(signatures)[flat_mask]
        # find number of kmers to keep per allele
        mask_by_alleles = nps.RaggedArray(flat_mask, ak.to_numpy(ak.num(ak.flatten(signatures))))
        n_per_allele = np.sum(mask_by_alleles, axis=1)
        new = ak.unflatten(keep, n_per_allele)
        n_alleles_per_variant = ak.num(signatures)
        new = ak.unflatten(new, n_alleles_per_variant)
        #logging.info("Done filtering")
        self.signatures = new
        """
        self.signatures = signatures  # new sorting
        self.filter(flat_mask)

    def filter(self, flat_mask: np.ndarray):
        """
        Replaces signatures by keeping only those in keep.
        flat_mask : flat np array where True are signatures to keep
        """
        signatures = self.signatures
        keep = ak.ravel(self.signatures)[flat_mask]
        # find number of kmers to keep per allele
        mask_by_alleles = nps.RaggedArray(flat_mask, ak.to_numpy(ak.num(ak.flatten(signatures))))
        n_per_allele = np.sum(mask_by_alleles, axis=1)
        new = ak.unflatten(keep, n_per_allele)
        n_alleles_per_variant = ak.num(signatures)
        new = ak.unflatten(new, n_alleles_per_variant)
        # logging.info("Done filtering")
        self.signatures = new

    def get_as_flat_kmers(self, node_mapping: VariantAlleleToNodeMap):
        signatures = self.signatures

        all_kmers = []
        all_node_ids = []

        flat_kmers = ak.to_numpy(ak.ravel(signatures)).astype(np.uint64)
        assert flat_kmers.dtype == np.uint64
        flat_node_ids = np.zeros_like(flat_kmers, dtype=np.int32)
        node_mapping = node_mapping.node_ids.ravel()  # ragged array of variants x alleles

        @numba.jit(nopython=True)
        def get_node_ids(signatures, flat_node_ids, node_mapping):
            allele_number = 0  # index in flat node_mapping array, increases for each allele
            i = 0
            for variant in range(len(signatures)):
                for allele in range(len(signatures[variant])):
                    for kmer in range(len(signatures[variant][allele])):
                        node_id = node_mapping[allele_number]
                        flat_node_ids[i] = node_id
                        i += 1
                    allele_number += 1

        get_node_ids(signatures, flat_node_ids, node_mapping)
        return FlatKmers(flat_kmers, flat_node_ids)


        # todo: vectorize
        for variant in range(len(signatures)):
            for allele in range(len(signatures[variant])):
                node_id = node_mapping.lookup(variant, allele)
                kmers = ak.to_numpy(signatures[variant, allele])
                all_kmers.append(kmers)
                all_node_ids.append(np.full(len(kmers), node_id))

        return FlatKmers(np.concatenate(all_kmers), np.concatenate(all_node_ids))

    def get_as_kmer_index(self, k, node_mapping: VariantAlleleToNodeMap, modulo=103, include_reverse_complements=True):
        flat = self.get_as_flat_kmers(node_mapping)
        if include_reverse_complements:
            rev_comp_flat = flat.get_reverse_complement_flat_kmers(k)
            flat = FlatKmers.from_multiple_flat_kmers([flat, rev_comp_flat])

        assert modulo > len(flat._hashes), "Modulo is too small for number of kmers. Increase"
        index = KmerIndex.from_flat_kmers(flat, skip_frequencies=True, modulo=modulo)
        index.convert_to_int32()
        return index

    def remove_too_frequent_signatures(self, scorer, threshold=100):
        """
        Removes signatures that have score above threshold
        """
        flat_signatures = ak.to_numpy(ak.ravel(self.signatures))
        scores = scorer[flat_signatures]
        mask = scores > threshold
        logging.info(f"{np.sum(mask)}/{len(flat_signatures)} signatures removed because they had score above {threshold}.")
        self.filter(~mask)


class SignatureFinder:
    """
    For each variant chooses a signature (a set of kmers) based one a Scores
    """
    def __init__(self, paths: Paths, scorer: Scorer = None, k=3):
        self._paths = paths
        self._scorer = scorer
        self._k = k
        self._chosen_ref_kmers = []
        self._chosen_alt_kmers = []


class MultiAllelicSignatureFinder(SignatureFinder):
    def __init__(self, paths: Paths, scorer: Scorer = None, k=3):
        self._paths = paths
        self._scorer = scorer
        self._k = k
        self._chosen_kmers = []  # n_variants x n_alleles x kmers at allele
        t0 = time.perf_counter()
        self._kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(self._paths.paths, self._k)
        logging.info("Finding all kmers around variants took %.4f seconds" % (time.perf_counter() - t0))


    def run(self) -> Signatures:
        kmers = self._kmers
        kmer_matrix = kmers.kmers
        # kmers is n_paths x n_variants x n_windows. Window size is variable
        variant_alleles = self._paths.variant_alleles
        logging.info("Scoring kmers")
        t0 = time.perf_counter()
        kmer_scores = kmers.score_kmers(self._scorer)
        logging.info("Scoring kmers took %.4f seconds" % (time.perf_counter() - t0))


        # for each allele find the best scoring window position where no kmers are found in other alleles
        for i, variant in enumerate(range(kmers.n_variants)):
            # find all unique kmers, but only count kmer once if it is at the
            # same window at the same allele
            alleles = variant_alleles[:,variant]
            unique_alleles = np.unique(alleles)

            # Finding all unique kmers over all alleles
            all_unique_kmers = []
            for allele in unique_alleles:
                rows = np.where(alleles == allele)[0]
                n_windows = len(kmer_matrix[rows[0], variant])  # all should have same window size (same allele)
                assert n_windows > 0
                for window in range(n_windows):
                    all_unique_kmers.append(np.unique(kmer_matrix[rows, variant, window]))

            #all_unique, all_counts = np.unique(np.concatenate(all_unique_kmers), return_counts=True)

            # for each allele, find windows where all kmers only exist once in all_unique
            kmers_found_at_variant = []
            for allele in range(np.max(unique_alleles)+1):
                paths = np.where(alleles == allele)[0] # which paths have this allele
                assert len(paths) > 0, "No paths with allele %d" % allele
                n_windows = len(kmer_matrix[paths[0], variant])  # all should have same window size (same allele)
                assert n_windows > 0
                window_scores = ak.to_numpy(np.min(kmer_scores[paths, variant], axis=1))
                # add 1 to last score to choose last on tie
                window_scores[-1] += 1
                best_kmers = np.unique(kmer_matrix[paths, variant,
                    np.argmax(window_scores)])

                #for window in range(min(n_windows, 5)):
                #    potential = kmer_matrix[paths, variant, window]
                #    if all(all_counts[all_unique == kmer] == 1 for kmer in potential):
                #        #score = np.min(self._scorer.score_kmers(potential))  # worst score of all kmers
                #        score = np.min(kmer_scores[paths, variant, window])
                #        if score > best_score:
                #            best_kmers = np.unique(potential)
                #            best_score = score

                kmers_found_at_variant.append(best_kmers)

            self._chosen_kmers.append(ak.Array(kmers_found_at_variant))

        return MultiAllelicSignatures(ak.Array(self._chosen_kmers))


class MultiAllelicSignatureFinderV2(SignatureFinder):
    """
    New version that uses VariantWindowKmers2 to get vectorized scoring
    """
    def __init__(self, variant_window_kmers: 'VariantWindowKmers2', scorer: Scorer, k=3, sv_min_size=40):
        """
        NOTE: sv_min_size referes to the minimum kmers for a variant to be considered a SV,
        which will trigger more through kmer finding. Number of kmers might be much
        lower than number of base pairs if spacinng has been used when selecting kmers
        """
        self._scorer = scorer
        self._k = k
        self._chosen_kmers = []  # n_variants x n_alleles x kmers at allele
        t0 = time.perf_counter()
        #self._kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(paths, self._k)
        #self._kmers = VariantWindowKmers2.from_matrix_variant_window_kmers(self._kmers, paths.variant_alleles.matrix)
        self._kmers = variant_window_kmers
        # first replace nonuniqe kmers with something
        self._replace_nonunique_with = -1
        #self._kmers.replace_nonunique_kmers(self._replace_nonunique_with)
        self._signatures_found = None
        self._sv_min_size = sv_min_size

    def _score_signatures(self, add_dummy_count_to_index=-1):
        scores = self._kmers.score_kmers(self._scorer)
        t0 = time.perf_counter()
        log_memory_usage_now("Before summing scores")
        window_scores = np.sum(scores, axis=2)  # or np.min?
        log_memory_usage_now("After summing scores")
        #logging.info("Summing scores took %.4f sec" % (time.perf_counter() - t0))
        # add 1 in score to the last window, so that is preferred on a tie. Convert to RaggedArray to be able to add
        # hackish flatten and unflatten

        t0 = time.perf_counter()
        window_scores_tmp = nps.RaggedArray(ak.to_numpy(ak.flatten(ak.flatten(window_scores))).astype(float),
                                            ak.to_numpy(ak.num(ak.flatten(window_scores))))
        window_scores_tmp[:, add_dummy_count_to_index] += 0.1
        self._window_scores = ak.unflatten(ak.unflatten(window_scores_tmp.ravel(), ak.num(ak.flatten(window_scores))),
                                     ak.num(window_scores))
        #logging.info("Postprocessing scores took %.4f sec" % (time.perf_counter() - t0))

    def run(self, add_dummy_count_to_index=-1) -> MultiAllelicSignatures:
        # idea is to first find the score for each window position at each allele at each variant
        # as the worst window score
        # Then, best window can be found using argmax over these scores
        tstart = time.perf_counter()
        self._score_signatures(add_dummy_count_to_index=add_dummy_count_to_index)
        #logging.info("Scoring signatures took %.4f sec" % (time.perf_counter() - tstart))
        best_windows = np.argmax(self._window_scores, axis=-1)
        # create a best windows array with same length as number of paths on all alleles
        kmers = self._kmers.kmers
        # number of paths per allele
        lengths = ak.to_numpy(ak.num(ak.flatten(kmers)))
        # hacky way to make mask: Create ragged array of the size we want, fill
        best_windows_mask_ones = nps.RaggedArray(np.ones(np.sum(lengths), dtype=int), lengths)
        best_windows_mask = best_windows_mask_ones * ak.to_numpy(ak.flatten(best_windows)).data[:, None]
        best_windows_mask_flat = best_windows_mask.ravel().astype(int)

        # flat kmers is structured by windows
        # first all windows on allele 0, then allele 1, then next variant etc
        flat_kmers = ak.flatten(ak.flatten(kmers))
        chosen_kmers_flat = flat_kmers[np.arange(len(flat_kmers)), best_windows_mask_flat]

        # restructure into variant x allele x kmers
        n_paths_per_allele = ak.num(ak.flatten(kmers))
        chosen_kmers_by_allele = ak.unflatten(chosen_kmers_flat, n_paths_per_allele)

        # restructure into variants
        n_alleles_per_variant = ak.num(kmers)
        chosen_kmers_by_variants = ak.unflatten(chosen_kmers_by_allele, n_alleles_per_variant)

        signatures = MultiAllelicSignatures(chosen_kmers_by_variants)
        #logging.info("Finding signatures took %.4f sec" % (time.perf_counter() - tstart))
        #logging.info("Removing nonunique")
        t0 = time.perf_counter()
        signatures.filter_nonunique_on_alleles(also_remove=self._replace_nonunique_with)
        #logging.info("Removing nonunique took %.4f sec" % (time.perf_counter() - t0))

        self._signatures_found = signatures
        t0 = time.perf_counter()
        self._manually_process_svs()
        #logging.info("Manually processing SVs took %.4f sec" % (time.perf_counter() - t0))

        return self._signatures_found

    def _manually_process_svs(self):
        kmers = self._kmers.kmers
        signatures = self._signatures_found.signatures
        scores = self._window_scores
        is_sv = self._kmers.get_mask_of_svs(self._sv_min_size)

        # make new MultiAllelicSignatures for single variants thare are to be changed
        # concatenate results (since ak array is immutable)
        changed = {}  # variant_id -> MultiAllelicSignatures
        for sv_id in np.where(is_sv)[0]:
            # check if sv has nonunique kmers
            all_kmers = ak.ravel(signatures[sv_id, :, :])
            if len(np.unique(all_kmers)) != len(all_kmers):

                # for each allele, find the kmer with best score that is not
                # in any other allele
                sv_kmers = kmers[sv_id]
                sv_scores = scores[sv_id]

                t0 = time.perf_counter()
                to_add = MultiAllelicSignatureFinderV2.manually_find_kmers(sv_kmers, sv_scores)
                time_spent = time.perf_counter()-t0
                if time_spent > 0.1:
                    # sv took a lot of time to index, log debug
                    logging.debug(f"Sv {sv_id} with {len(signatures[sv_id])} allele and {len(sv_kmers[0][0])} windows took {time_spent} sec")

                changed[sv_id] = to_add

        to_concatenate = []
        prev_id = 0
        for sv_id, sig in changed.items():
            to_concatenate.append(signatures[prev_id:sv_id])
            to_concatenate.append(sig.signatures)
            prev_id = sv_id + 1
        to_concatenate.append(signatures[prev_id:])
        new = np.concatenate(to_concatenate)

        assert len(new) == len(signatures)
        self._signatures_found = MultiAllelicSignatures(new)

    @staticmethod
    def manually_find_kmers(kmers: ak.Array, scores: ak.Array, force=True) -> MultiAllelicSignatures:
        """
        This method "manually" finds best kmers by searching more thoroughly. To be used for tricky large variants where there are few unique kmers.
        Kmers and scores are ak.Arrays of shape n_alleles x n_paths x n_windows

        if force = True, kmers will always be returned
        if False, will not give any kmers if only bad kmers are found
        """

        # make a kmer scorer counting only kmers on this variant
        # get scores for kmers, add those scores to the existing scores
        # idea is to try to find unique kmers within this variant
        local_modulo = 1000001
        local_scorer = np.zeros(local_modulo, dtype=float)
        all_variant_kmers = ak.to_numpy(ak.ravel(kmers))
        local_scorer += np.bincount((all_variant_kmers % local_modulo).astype(int), minlength=local_modulo)

        n_alleles = len(kmers)

        def _find(kmers, local_scorer):
            new_variant_kmers = []
            n_alleles = len(kmers)
            #n_paths = len(kmers[0])  # same on all alleles
            for allele in range(n_alleles):
                #t0 = time.perf_counter()
                allele_kmers = ak.to_numpy(kmers[allele])  # x n_paths x n_windows
                n_paths = len(allele_kmers)
                n_windows = len(allele_kmers[0])  # same windows on all paths
                # pick some window locations
                window_locations = range(0, n_windows, max(1, n_windows // 100))

                window_kmers_matrix = allele_kmers[:, window_locations]
                scores_for_windows = ak.to_numpy(scores[allele, window_locations])
                local_scores_for_windows = np.max(
                    local_scorer[window_kmers_matrix.ravel() % local_modulo].reshape(window_kmers_matrix.shape), axis=0
                )

                scores_for_windows -= local_scores_for_windows * 3  # weigh local scores more than global scores. More important to find locally unique kmers
                best_window = window_locations[np.argmax(scores_for_windows)]
                chosen = np.unique(ak.to_numpy(kmers[allele, :, best_window]))
                assert chosen.dtype == np.uint64
                new_variant_kmers.append(chosen)

            return new_variant_kmers

        new_variant_kmers = _find(kmers, local_scorer)

        # hacky way to build ak array to keep dtype
        # awkard array changes dtype when making from list
        flat = np.concatenate(new_variant_kmers)
        shape = [len(n) for n in new_variant_kmers]
        return MultiAllelicSignatures(ak.unflatten(ak.unflatten(flat, shape), [n_alleles]))


class SignatureFinder2(SignatureFinder):
    def run(self) -> Signatures:
        # for each path, first creates a RaggedArray with windows for each variant
        paths = MatrixVariantWindowKmers.from_paths(self._paths.paths, self._k)  # ._paths.get_windows_around_variants(self._k)
        chosen_ref_kmers = []
        chosen_alt_kmers = []

        for variant in tqdm.tqdm(range(self._paths.n_variants()), desc="Finding signatures", unit="variants",
                                 total=self._paths.n_variants()):
            best_kmers = None
            best_score = -10000000000000000000
            all_ref_kmers = paths.get_kmers(variant, 0, self._paths.variant_alleles)
            all_alt_kmers = paths.get_kmers(variant, 1, self._paths.variant_alleles)
            #assert np.all(all_ref_kmers.shape[1] == all_alt_kmers.shape[1])

            for window in range(len(all_ref_kmers[0])):
                ref_kmers = np.unique(all_ref_kmers[:, window])  # .raw().ravel())
                alt_kmers = np.unique(all_alt_kmers[:, window])  # .raw().ravel())

                all_kmers = np.concatenate([ref_kmers, alt_kmers])
                window_score = self._scorer.score_kmers(all_kmers) if self._scorer is not None else 0

                # if overlap between ref and alt, lower score
                if len(set(ref_kmers).intersection(alt_kmers)) > 0:
                    window_score -= 1000

                if window_score > best_score:
                    best_kmers = [ref_kmers, alt_kmers]
                    best_score = window_score

            chosen_ref_kmers.append(best_kmers[0])
            chosen_alt_kmers.append(best_kmers[1])

        return Signatures(nps.RaggedArray(chosen_ref_kmers), nps.RaggedArray(chosen_alt_kmers))


class SignatureFinder3(SignatureFinder):
    def run(self, variants: Variants = None) -> Signatures:
        variant_alleles = self._paths.variant_alleles
        kmers = MatrixVariantWindowKmers.from_paths(self._paths.paths, self._k)
        self._all_possible_kmers = kmers
        best_kmers = kmers.get_best_kmers(self._scorer)

        # hack to split matrix into ref and alt kmers
        n_variants = self._paths.n_variants()
        n_paths = len(self._paths.paths)
        all_ref_kmers = best_kmers.T[variant_alleles.T == 0].reshape((n_variants, n_paths//2))
        all_alt_kmers = best_kmers.T[variant_alleles.T == 1].reshape((n_variants, n_paths//2))

        n_snps_with_identical_kmers = 0
        n_indels_with_identical_kmers = 0
        n_snps_total = 0
        n_indels_total = 0

        indels_with_identical_kmers = []

        for variant_id, (ref_kmers, alt_kmers) in tqdm.tqdm(enumerate(zip(all_ref_kmers, all_alt_kmers)), desc="Iterating signatures", unit="variants", total=n_variants):
            ref_kmers = np.unique(ref_kmers)
            alt_kmers = np.unique(alt_kmers)

            if variants is not None:
                is_snp = len(variants.ref_seq[variant_id]) == 1 and len(variants.alt_seq[variant_id]) == 1
                if is_snp:
                    n_snps_total += 1
                else:
                    n_indels_total += 1

            if set(ref_kmers).intersection(alt_kmers):
                ref_kmers = np.array([])
                alt_kmers = np.array([])

                if variants is not None:
                    if is_snp:
                        n_snps_with_identical_kmers += 1
                    else:
                        #print("Indel with identical kmers, sequences:")
                        #print(variants.ref_seq[variant_id], variants.alt_seq[variant_id])
                        n_indels_with_identical_kmers += 1
                        indels_with_identical_kmers.append(variant_id)

            self._chosen_ref_kmers.append(ref_kmers)
            self._chosen_alt_kmers.append(alt_kmers)

        if variants is not None and False:
            self._manuall_process_indels(indels_with_identical_kmers)
            #self._manually_process_svs(variants)

        logging.info(f"{n_snps_with_identical_kmers}/{n_snps_total} snps had identical kmers in ref and alt")
        logging.info(f"{n_indels_with_identical_kmers}/{n_indels_total} indels had identical kmers in ref and alt")

        return Signatures(nps.RaggedArray(self._chosen_ref_kmers), nps.RaggedArray(self._chosen_alt_kmers))

    def _manuall_process_indels(self, indel_ids):
        # try to find kmers that are unique to the alleles
        n_ref_replaced = 0
        n_alt_replaced = 0



        for id in indel_ids:
            options_ref = self._all_possible_kmers.get_kmers(id, 0)
            options_alt = self._all_possible_kmers.get_kmers(id, 1)

            window_scores_ref = []  # score for each window
            window_scores_alt = []

            # all windows for ref/alt until now, and score them
            window_kmers_ref = [k for k in options_ref.T]
            window_kmers_alt = [k for k in options_alt.T]

            assert len(window_kmers_alt) == len(window_kmers_ref)

            unique_ref = np.unique(options_ref)
            unique_alt = np.unique(options_alt)
            print("Unique ref", unique_ref)
            print("Unique alt", unique_alt)

            # try to find a window position where no kmers are found in the other
            for window in range(len(window_kmers_ref)):
                ref_kmers = window_kmers_ref[window]
                alt_kmers = window_kmers_alt[window]
                print("Window", window, ":" , ref_kmers, alt_kmers)

                score_ref = np.min(self._scorer.score_kmers(ref_kmers))  # wors score of the score of kmers in this window
                score_alt = np.min(self._scorer.score_kmers(alt_kmers))

                # if any kmers are found in the other, decrease score
                if len(set(ref_kmers).intersection(unique_alt)) > 0:
                    score_ref -= 1000
                if len(set(alt_kmers).intersection(unique_ref)) > 0:
                    score_alt -= 1000

                window_scores_ref.append(score_ref)
                window_scores_alt.append(score_alt)

            assert len(window_scores_ref) == len(window_kmers_ref)

            # pick best window for ref/alt
            print("WIndow scores")
            print(window_scores_ref)
            print(window_scores_alt)
            best_ref = np.argmax(window_scores_ref)
            best_alt = np.argmax(window_scores_alt)
            print(f"Picked window {best_ref} for ref and {best_alt} for alt with scores {window_scores_ref[best_ref]} and {window_scores_alt[best_alt]}")
            # only replace kmers if we found some unique
            if window_scores_ref[best_ref] > -1000 and window_scores_alt[best_alt] > -1000:
                self._chosen_ref_kmers[id] = np.unique(window_kmers_ref[best_ref])
                self._chosen_alt_kmers[id] = np.unique(window_kmers_alt[best_alt])
                n_alt_replaced += 1
                n_ref_replaced += 1

        print(f"Replaced {n_ref_replaced} ref and {n_alt_replaced} alt kmers for indels")

    def _manually_process_svs(self, variants: Variants):
        # slower version that manually tries to pick better signatures for each sv
        # returns a Signature object where  that can be merged with another Signature object
        is_sv = np.where((variants.ref_seq.shape[1] > self._k) | (variants.alt_seq.shape[1] > self._k))[0]
        logging.info("N large insertions: ", np.sum(variants.alt_seq.shape[1] > self._k))
        logging.info("N large deletions: ", np.sum(variants.ref_seq.shape[1] > self._k))
        logging.info(f"There are {len(is_sv)} SVs that signatures will be adjusted for")
        n_ref_replaced = 0
        n_alt_replaced = 0

        for sv_id in is_sv:
            variant = variants[sv_id]
            current_options_ref = np.unique(self._all_possible_kmers.get_kmers(sv_id, 0))
            current_options_alt = np.unique(self._all_possible_kmers.get_kmers(sv_id, 1))

            # add kmers inside nodes to possible options
            new_ref = bnp.get_kmers(variant.ref_seq.ravel(), self._k).ravel().raw().astype(np.uint64)
            new_alt = bnp.get_kmers(variant.alt_seq.ravel(), self._k).ravel().raw().astype(np.uint64)

            options_ref = np.concatenate([current_options_ref, new_ref])
            options_alt = np.concatenate([current_options_alt, new_alt])

            # remove kmers that are in both ref and alt
            unique_options_ref = np.setdiff1d(options_ref, options_alt)
            unique_options_alt = np.setdiff1d(options_alt, options_ref)

            #print("Unique options")
            #print(unique_options_ref, unique_options_alt)

            # pick the option with best score
            if len(unique_options_ref) > 0 and len(unique_options_alt) > 0:
                score_ref = self._scorer.score_kmers(unique_options_ref)
                #print("Best score ref: ", np.max(score_ref))
                best_ref = unique_options_ref[np.argmax(score_ref)]
                self._chosen_ref_kmers[sv_id] = np.array([best_ref])
                n_ref_replaced += 1

            #if len(unique_options_alt) > 0:
                score_alt = self._scorer.score_kmers(unique_options_alt)
                #print("Best score alt: ", np.max(score_alt))
                best_alt = unique_options_alt[np.argmax(score_alt)]
                self._chosen_alt_kmers[sv_id] = np.array([best_alt])
                n_alt_replaced += 1

            # alternatively, one could pick more than one good kmer if the allele is big

        print(f"Replaced {n_ref_replaced} ref and {n_alt_replaced} alt kmers for SVs")


@dataclass
class VariantWindowKmers2:
    """
    Similar to MatrixVariantWindowKmers, but paths is not a dimension, instead there can be multiple
    entries per allele to represent the possibilities
    """
    kmers: ak.Array  # n_variants x n_alleles x n_paths_on_allele x n_windows

    def get_mask_of_svs(self, sv_min_window_size=10):
        # svs are variants where any allele has a path with many windows
        return ak.to_numpy(np.any(np.any(ak.num(self.kmers, axis=3) > sv_min_window_size, axis=-1), axis=-1))


    def __eq__(self, other):
        return np.all(self.kmers == other.kmers)

    @classmethod
    def from_list(cls, l) -> 'VariantWindowKmers2':
        return cls(ak.Array(l))

    def to_kmer_list(self, k):
        return [
                [
                    [
                        [
                            kmer_hash_to_sequence(window_kmer, k) for window_kmer in window_kmers
                        ]
                        for window_kmers in path_kmers
                    ]
                    for path_kmers in allele_kmers
                ]
                for allele_kmers in self.kmers
            ]

    @property
    def n_variants(self):
        return len(self.kmers[0])

    def replace_nonunique_kmers(self, replace_with=0):
        #raise NotImplementedError("Not implemented")
        # filter so that there are only unique kmers at each window position on each allele
        sorted_kmers = np.sort(self.kmers, axis=2)

        mask = np.ones_like(ak.to_numpy(ak.ravel(sorted_kmers)), dtype=bool)  # which kmers to keep

        @numba.jit(nopython=True)
        def create_mask(flat_mask, signatures):
            flat_index = 0
            for variant in range(len(signatures)):
                for allele in range(len(signatures[variant])):
                    n_windows = len(signatures[variant][allele][0])  # window size is the same on all paths
                    n_paths = len(signatures[variant][allele])
                    for window in range(n_windows):
                        prev_kmer = np.uint64(0)
                        for path in range(n_paths):
                            flat_mask_index = flat_index + (path*n_windows) + window
                            kmer = signatures[variant][allele][path][window]
                            if kmer == prev_kmer and path > 0:
                                flat_mask[flat_mask_index] = False
                            prev_kmer = kmer
                    flat_index += n_paths * n_windows

        create_mask(mask, sorted_kmers)
        logging.info("Removed %d non-unique kmers" % np.sum(~mask))

        # use mask to create a new data structure
        flat_kmers = ak.to_numpy(ak.ravel(sorted_kmers))
        flat_kmers[~mask] = replace_with

        self.kmers = ak.unflatten(
                ak.unflatten(
                    ak.unflatten(flat_kmers, ak.num(ak.flatten(ak.flatten(self.kmers)))),
                ak.num(ak.flatten(self.kmers))),
            ak.num(self.kmers))

    def score_kmers(self, scorer):
        # returns an ak.Array of the same shape with kmer scores
        # hackish way to flatten/unflatten since awkard array doesn't support unravel (only flatten/unflatten on single dimension)
        flat = ak.to_numpy(ak.flatten(ak.flatten(ak.flatten(self.kmers))))
        t0 = time.perf_counter()
        scores = scorer.score_kmers(flat)
        #logging.info(f"Scoring {len(flat)} kmers took {time.perf_counter() - t0} sec")
        # always score 0 as 0, tmp hack
        #scores[flat == 0] = 0
        a = self.kmers
        t0 = time.perf_counter()
        scores = ak.unflatten(ak.unflatten(ak.unflatten(scores, ak.num(ak.flatten(ak.flatten(a)))), ak.num(ak.flatten(a))), ak.num(a))
        #logging.info("Unflatten took %.4f sec" % (time.perf_counter() - t0))
        return scores

    @classmethod
    def from_matrix_variant_window_kmers(cls, kmers: 'MatrixVariantWindowKmers', path_alleles: np.ndarray) -> 'MatrixVariantWindowKmers2':
        # make a flat data structure and unflatten it
        #print("Memory usage from_matrix_variant_window_kmer: %d" % get_memory_usage())
        kmers = kmers.kmers
        n_paths = path_alleles.shape[0]
        n_variants = path_alleles.shape[1]

        # indexes to use when putting all kmers into new flat structure
        # argsort trick: get correct order of kmers (sorted by allele)
        sorted_alleles = np.argsort(path_alleles, axis=0)
        # get indexes by sorted alleles columnwise
        flat_indexes = sorted_alleles.T.ravel() + n_paths * (np.arange(n_paths*n_variants)//n_paths)  # increase by n_paths each time

        #indexes in flat kmers should give kmers column wise from original data structure (columns are variants)
        kmer_indexes = np.arange(0, n_paths*n_variants).reshape(n_paths, n_variants).T.ravel()
        kmers_reshaped = ak.flatten(kmers)[kmer_indexes][flat_indexes]
        window_structure = ak.num(kmers_reshaped)
        kmers_reshaped_flat = ak.flatten(kmers_reshaped)
        flat = kmers_reshaped_flat

        # Unflatten back to correct structure (variants x alleles x n_paths on allele x window kmers)
        # group by window
        grouped_by_window = ak.unflatten(flat, window_structure)

        # group by allele
        n_per_allele = ak.to_numpy(ak.ravel(ak.run_lengths(np.sort(path_alleles, axis=0).T)))
        grouped_by_alleles = ak.unflatten(grouped_by_window, n_per_allele)

        # group by variant
        # note: This is wrong if some alleles are not covered by paths. This assume every allele is covered by at least one path
        n_alleles_per_variant = np.max(path_alleles, axis=0) + 1
        #logging.info("Total alleles: %d" % np.sum(n_alleles_per_variant))
        #logging.info("Max alleles on a variant: %d" % np.max(n_alleles_per_variant))
        if np.max(n_alleles_per_variant) > n_paths:
            logging.warning("There are not enough paths to cover all alleles on variants. Results may be weird")

        #logging.info("Total signatures alleles: %d" % len(grouped_by_alleles))
        grouped_by_variants = ak.unflatten(grouped_by_alleles, n_alleles_per_variant)
        return cls(grouped_by_variants)


    def describe(self, k):
        from graph_kmer_index import kmer_hash_to_sequence
        for variant_id, variant in enumerate(self.kmers):
            print("Variant %d" % variant_id)
            for allele_id, allele in enumerate(variant):
                print("  Allele %d" % allele_id)
                for path_id, path in enumerate(allele):
                    print("    Path: " + " ".join([kmer_hash_to_sequence(kmer, k) for kmer in path]))

@dataclass
class MatrixVariantWindowKmers:
    """
    Represents kmers around variants in a 3-dimensional matrix, possible when n kmers per variant allele is fixed
    """
    kmers: Union[np.ndarray, ak.Array]  # n_paths, n variants x n_windows.
    # Can be numpy matrix if even window sizes and fixed number of alleles

    def describe(self, k):
        from graph_kmer_index import kmer_hash_to_sequence
        for path_id, path in enumerate(self.kmers):
            print("Path %d" % path_id)
            for variant_id, variant in enumerate(path):
                print("  Variant %d: %s" % (variant_id, ' '.join([kmer_hash_to_sequence(kmer, k) for kmer in variant])))

    @property
    def n_variants(self):
        if isinstance(self.kmers, ak.Array):
            return len(self.kmers[0])
        else:
            return self.kmers.shape[1]

    @classmethod
    def from_paths(cls, path_sequences: PathSequences, k):
        n_variants = path_sequences.n_variants()
        n_paths = len(path_sequences)
        n_windows = k - 1 - k//2
        matrix = np.zeros((n_paths, n_variants, n_windows), dtype=np.uint64)

        for i, path in enumerate(path_sequences.iter_path_sequences()):
            starts = path._shape.starts[1::2]
            window_starts = starts - k + 1 + k//2
            #window_ends = np.minimum(starts + k, path.size)
            window_ends = starts + k - 1
            windows = bnp.ragged_slice(path.ravel(), window_starts, window_ends)
            kmers = bnp.get_kmers(windows, k=k).ravel().raw().astype(np.uint64)
            kmers = kmers.reshape((n_variants, n_windows))
            matrix[i, :, :] = kmers

        return cls(matrix)

    @classmethod
    def from_paths_with_flexible_window_size(cls,
                                             path_sequences: PathSequences,
                                             k,
                                             spacing=0,
                                             only_pick_kmers_inside_big_alleles=False,
                                             minimum_overlap_with_variant=1):
        """
        IF only_pick_kmers_inside_big_alleles is True, then for big alleles, kmers will only be chosen so that they are inside
        the allele
        minimum_overlap_with_variant is minimum number of base pairs a kmer needs to overlap with variant allele
        Should be at least 2 if indels have flanking bases
        """
        # uses different windows for each variant, stores in an awkward array
        kmers_found = []  # one RaggedArray for each path. Each element in ragged array represents a variant allele

        outer_dimension = []
        inner_dimensions = []
        t_get_kmers = 0
        t_div = 0
        t_div2 = 0

        for i, path in enumerate(path_sequences.iter_path_sequences()):
            t0 = time.perf_counter()
            variant_sizes = path[1::2].shape[1]
            window_sizes = variant_sizes + (k-minimum_overlap_with_variant) + (k-1)  # left part outside + right part outside
            n_kmers_in_each_window = window_sizes - k + 1
            assert np.all(n_kmers_in_each_window > 0)
            # for each variant, find the kmers for this path and fill into the window
            starts = path._shape.starts[1::2]
            window_starts = starts - k + minimum_overlap_with_variant

            if only_pick_kmers_inside_big_alleles:
                window_starts[variant_sizes > k] += (k-minimum_overlap_with_variant)
                window_sizes[variant_sizes > k] -= (k-minimum_overlap_with_variant + k-1)

            window_ends = window_starts + window_sizes
            assert np.all(window_ends > window_starts)
            assert np.all(window_ends <= len(path.ravel())), "Window end %d > length of path %d" % (np.max(window_ends), len(path.ravel()))
            #logging.info("DTYPE window starts/ends: %s/%s" % (window_starts.dtype, window_ends.dtype))
            windows = bnp.ragged_slice(path.ravel(), window_starts, window_ends)
            assert np.all(window_sizes == windows.shape[1]), "%s != %s" % (window_sizes, windows.shape[1])
            t_div += time.perf_counter() - t0

            t0 = time.perf_counter()
            ragged_kmers = bnp.get_kmers(windows, k=k)
            # subsample kmers?
            if spacing > 1:
                ragged_kmers = ragged_kmers[:, ::spacing]


            kmers = ragged_kmers.ravel().raw().astype(np.uint64)
            t_get_kmers += time.perf_counter() - t0
            #assert len(kmers) == np.sum(n_kmers_in_each_window)
            assert np.all(n_kmers_in_each_window > 0)

            t0 = time.perf_counter()
            kmers = nps.RaggedArray(kmers, ragged_kmers.shape)
            inner_dimensions.append(ragged_kmers.shape[1])  #kmers.shape[1])
            outer_dimension.append(len(kmers))
            kmers_found.append(
                kmers.ravel()
            )
            t_div2 += time.perf_counter() - t0

        #logging.info("Getting kmers took %.4f seconds. Time div: %.4f. Time div2: %.4f" % (t_get_kmers, t_div, t_div2))

        t0 = time.perf_counter()
        flat_kmers = np.concatenate(kmers_found)
        assert flat_kmers.dtype == np.uint64
        kmers_found = ak.unflatten(flat_kmers, np.concatenate(inner_dimensions))
        kmers_found = ak.unflatten(kmers_found, outer_dimension)
        #logging.info("Unflattening took %.4f seconds" % (time.perf_counter() - t0))
        return cls(ak.Array(kmers_found))  #matrix[i, :, :] = kmers

    def get_kmers(self, variant, allele, variant_alleles):
        relevant_paths = np.where(variant_alleles[:, variant] == allele)[0]
        kmers = self.kmers[relevant_paths, variant, :]
        if isinstance(kmers, ak.Array):
            return ak.to_numpy(kmers)
        else:
            return kmers

    def get_best_kmers(self, scorer):
        # score all kmers
        # returns a matrix of shape n_paths x n_variants containing the kmers chosen for each variant
        # pick one window per variant
        #logging.info("Scoring all candidate kmers")
        scores = scorer.score_kmers(self.kmers.ravel()).reshape(self.kmers.shape)
        # window scores are the lowest score among all paths in windows
        #scores = np.min(scores, axis=0)
        scores = np.sum(scores, axis=0)
        # score rightmost window a bit more to prefer this on a tie, good for indels
        scores[:, -1] += 1
        # best window has the highest score
        #logging.info("Finding best windows based on scores")
        best_windows = np.argmax(scores, axis=1)
        best_kmers = self.kmers[:, np.arange(self.kmers.shape[1]), best_windows]
        assert best_kmers.shape[0] == self.kmers.shape[0]
        return best_kmers

    def score_kmers(self, scorer):
        # Returns an ak array of same shape with scores
        assert isinstance(self.kmers, ak.Array), "Only supported for ak array now"
        new = []
        for path_kmers in self.kmers:
            flat = ak.to_numpy(ak.ravel(path_kmers))
            scores = scorer.score_kmers(flat)
            scores = ak.unflatten(scores, ak.num(path_kmers))
            new.append(scores)

        return ak.Array(new)


def get_signatures(k: int, paths: Paths, scorer, chunk_size=1000, add_dummy_count_to_index=-1, spacing=0,
                   minimum_overlap_with_variant=1):
    """Wrapper function that finds multiallelic signatures from paths"""
    log_memory_usage_now("Before MatrixVariantWindowKmers")

    # To keep max memory usage low, create signatures for chunks of variants, concatenate in the end
    n_variants = paths.n_variants()
    chunks = interval_chunks(0, n_variants, n_variants//chunk_size+1)
    all_signatures = []

    #todo make all subpaths before loop to avoid reading paths multiple times
    #all_subpaths = []
    #for from_variant, to_variant in tqdm.tqdm(chunks, desc="Making subpaths", unit="chunks", total=len(chunks)):
    #    all_subpaths.append(paths.subset_on_variants(from_variant, to_variant, k))

    all_subpaths = paths.chunk(chunks, padding=k)

    for i, (from_variant, to_variant) in enumerate(tqdm.tqdm(chunks, desc="Finding signatures", unit="chunks", total=len(chunks))):
        #logging.info("Subsetting paths on variants")
        #subpaths = paths.subset_on_variants(from_variant, to_variant, k)
        subpaths = all_subpaths[i]

        #logging.info("Making variant window kmers from paths (finding kmer candidates)")
        t0 = time.perf_counter()
        variant_window_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(
            subpaths.paths,
            k,
            spacing=spacing,
            only_pick_kmers_inside_big_alleles=True,
            minimum_overlap_with_variant=minimum_overlap_with_variant
        )
        #logging.info("Making signature kmers from paths took %.4f seconds" % (time.perf_counter() - t0))
        #log_memory_usage_now("After MatrixVariantWindowKmers")
        #logging.info("Converting variant window kmers to new data structure")
        #log_memory_usage_now("Before variant window kmers2")
        t0 = time.perf_counter()
        variant_window_kmers2 = VariantWindowKmers2.from_matrix_variant_window_kmers(variant_window_kmers,
                                                                                     subpaths.variant_alleles.matrix,)
        #logging.info("Conversion took %.4f seconds" % (time.perf_counter() - t0))
        #log_memory_usage_now("After variant window kmers2")
        #logging.info("Finding best signatures for variants")
        signatures = MultiAllelicSignatureFinderV2(variant_window_kmers2, scorer=scorer, k=k, sv_min_size=50//max(1, spacing)).run(add_dummy_count_to_index)
        # Removing frequent signatures is not necessary, but will speed up mapping model since more signatures are pruned from paths
        signatures.remove_too_frequent_signatures(scorer, 1000000)
        all_signatures.append(signatures)

    for s in all_subpaths:
        s.remove_tmp_files()

    return MultiAllelicSignatures.from_multiple(all_signatures)


def get_signatures(k: int, paths: Paths, scorer, chunk_size=10000, add_dummy_count_to_index=-1, spacing=0,
                   minimum_overlap_with_variant=1, n_threads=1):
    """Wrapper function that finds multiallelic signatures from paths"""
    log_memory_usage_now("Before MatrixVariantWindowKmers")

    # To keep max memory usage low, create signatures for chunks of variants, concatenate in the end
    n_variants = paths.n_variants()
    chunks = interval_chunks(0, n_variants, n_variants//chunk_size+1)
    logging.info("Will find signatures for %d chunks" % len(chunks))
    all_signatures = []

    all_subpaths = paths.chunk(chunks, padding=k)

    t0 = time.perf_counter()
    if n_threads == 1:
        for chunk_index in tqdm.tqdm(range(len(all_subpaths)), desc="Finding signatures", unit="chunks", total=len(chunks)):
            signatures = find_signatures_for_chunk(add_dummy_count_to_index, all_subpaths, chunk_index, k,
                                                   minimum_overlap_with_variant, scorer, spacing)
            all_signatures.append(signatures)
    else:
        ray.init(num_cpus=n_threads, ignore_reinit_error=True)
        all_subpaths = ray.put(all_subpaths)
        scorer = ray.put(scorer)
        for chunk_index in range(len(chunks)):
            signatures = find_signatures_for_chunk_wrapper.remote(add_dummy_count_to_index, all_subpaths, chunk_index,
                                                                  k, minimum_overlap_with_variant, scorer, spacing)
            all_signatures.append(signatures)

        log_memory_usage_now("Before ray.get in get_signatures")
        all_signatures = ray.get(all_signatures)
        log_memory_usage_now("After ray.get in get_signatures")
        ray.shutdown()

    logging.info("Finding signatures with %d threads took %.4f seconds" % (n_threads, time.perf_counter() - t0))

    #for s in all_subpaths:
    #    s.remove_tmp_files()

    return MultiAllelicSignatures.from_multiple(all_signatures)


@ray.remote
def find_signatures_for_chunk_wrapper(*params):  # wrapper for ray
    return find_signatures_for_chunk(*params)


def find_signatures_for_chunk(add_dummy_count_to_index, all_subpaths, chunk_index, k, minimum_overlap_with_variant,
                              scorer, spacing):
    #log_memory_usage_now("Signature loop start")
    subpaths = all_subpaths[chunk_index]
    t0 = time.perf_counter()
    variant_window_kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(
        subpaths.paths,
        k,
        spacing=spacing,
        only_pick_kmers_inside_big_alleles=True,
        minimum_overlap_with_variant=minimum_overlap_with_variant
    )
    #log_memory_usage_now("After variant window kmers")
    #logging.info("Making signature kmers from paths took %.4f seconds" % (time.perf_counter() - t0))
    t0 = time.perf_counter()
    variant_window_kmers2 = VariantWindowKmers2.from_matrix_variant_window_kmers(variant_window_kmers,
                                                                                 subpaths.variant_alleles.matrix, )
    #logging.info("Conversion took %.4f seconds" % (time.perf_counter() - t0))
    #log_memory_usage_now("After variant window kmers2")
    t0 = time.perf_counter()
    signatures = MultiAllelicSignatureFinderV2(variant_window_kmers2, scorer=scorer, k=k,
                                               sv_min_size=50 // max(1, spacing)).run(add_dummy_count_to_index)
    #logging.info("Finding best signatures for variants took %.4f seconds" % (time.perf_counter() - t0))
    t0 = time.perf_counter()
    # Removing frequent signatures is not necessary, but will speed up mapping model since more signatures are pruned from paths
    signatures.remove_too_frequent_signatures(scorer, 254)
    #logging.info("Removing frequent signatures took %.4f seconds" % (time.perf_counter() - t0))
    return signatures

