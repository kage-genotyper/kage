import logging
import time
from dataclasses import dataclass
import bionumpy as bnp
from graph_kmer_index import KmerIndex
import npstructures as nps
import numpy as np
import tqdm
from graph_kmer_index import FlatKmers, CollisionFreeKmerIndex
from .kmer_scoring import Scorer
from .paths import Paths, PathSequences
from ..preprocessing.variants import Variants, VariantAlleleToNodeMap
from typing import List, Union
import awkward as ak
from graph_kmer_index import FlatKmers


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
class MultialleleSignatures:
    signatures: ak.Array  # n_variants x n_alleles x signatures


@dataclass
class MultiAllelicSignatures:
    # there can be multiple signatures for one allele at a variant
    signatures: ak.Array  # n_variants x n_alleles x signatures

    @classmethod
    def from_list(cls, l) -> 'MultiAllelicSignatures':
       return cls(ak.Array(l))

    def get_as_flat_kmers(self, node_mapping: VariantAlleleToNodeMap):
        signatures = self.signatures

        all_kmers = []
        all_node_ids = []

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
            print("Variant", i)
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

            all_unique, all_counts = np.unique(np.concatenate(all_unique_kmers), return_counts=True)

            # for each allele, find windows where all kmers only exist once in all_unique
            kmers_found_at_variant = []
            for allele in range(np.max(unique_alleles)+1):
                paths = np.where(alleles == allele)[0] # which paths have this allele
                assert len(paths) > 0, "No paths with allele %d" % allele
                n_windows = len(kmer_matrix[paths[0], variant])  # all should have same window size (same allele)
                assert n_windows > 0
                best_score = -100000000000
                best_kmers = []
                best_kmers = np.unique(kmer_matrix[paths, variant,
                    np.argmax(np.min(kmer_scores[paths, variant], axis=1))])

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
    New version that uses MatrixVariantWindowKmers2 to get vectorized scoring
    """
    def __init__(self, variant_window_kmers: 'VariantWindowKmers2', scorer: Scorer, k=3):
        self._scorer = scorer
        self._k = k
        self._chosen_kmers = []  # n_variants x n_alleles x kmers at allele
        t0 = time.perf_counter()
        #self._kmers = MatrixVariantWindowKmers.from_paths_with_flexible_window_size(paths, self._k)
        #self._kmers = VariantWindowKmers2.from_matrix_variant_window_kmers(self._kmers, paths.variant_alleles.matrix)
        self._kmers = variant_window_kmers
        logging.info("Finding all kmers around variants took %.4f seconds" % (time.perf_counter() - t0))

    def run(self) -> MultiAllelicSignatures:
        # idea is to first find the score for each window position at each allele at each variant
        # as the wors window score
        # Then, best window can be found using argmax over these scores
        scores = self._kmers.score_kmers(self._scorer)
        window_scores = np.min(scores, axis=2)
        best_windows = np.argmax(window_scores, axis=-1)

        # create structure of variant x allele x kmers
        # todo: Should be possible to vectorize this
        signatures = []
        kmers = self._kmers.kmers
        n_variants = len(kmers)
        for variant in range(n_variants):
            variant_kmers = []
            for allele in range(len(kmers[variant])):
                variant_kmers.append(np.unique(kmers[variant, allele, :, best_windows[variant, allele]]))
            signatures.append(variant_kmers)

        return MultiAllelicSignatures.from_list(signatures)


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
                        print("Indel with identical kmers, sequences:")
                        print(variants.ref_seq[variant_id], variants.alt_seq[variant_id])
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
        print("N large insertions: ", np.sum(variants.alt_seq.shape[1] > self._k))
        print("N large deletions: ", np.sum(variants.ref_seq.shape[1] > self._k))
        print(f"There are {len(is_sv)} SVs that signatures will be adjusted for")
        n_ref_replaced = 0
        n_alt_replaced = 0

        for sv_id in is_sv:
            variant = variants[sv_id]
            current_options_ref = np.unique(self._all_possible_kmers.get_kmers(sv_id, 0))
            current_options_alt = np.unique(self._all_possible_kmers.get_kmers(sv_id, 1))

            #print("Ref/alt seq")
            #print(variant.ref_seq, variant.alt_seq)

            #print("current options")
            #print(current_options_ref, current_options_alt)

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

    @property
    def n_variants(self):
        return len(self.kmers[0])

    def score_kmers(self, scorer):
        # returns an ak.Array of the same shape with kmer scores
        # hackish way to flatten/unflatten since awkard array doesn't support unravel (only flatten/unflatten on single dimension)
        flat = ak.flatten(ak.flatten(ak.flatten(self.kmers)))
        scores = scorer.score_kmers(ak.to_numpy(flat))
        a = self.kmers
        scores = ak.unflatten(ak.unflatten(ak.unflatten(scores, ak.num(ak.flatten(ak.flatten(a)))), ak.num(ak.flatten(a))), ak.num(a))
        return scores

    @classmethod
    def from_matrix_variant_window_kmers(cls, kmers: 'MatrixVariantWindowKmers', path_alleles: np.ndarray) -> 'MatrixVariantWindowKmers2':
        new = []
        # todo: This is probably slow and can be vectorized
        n_alleles = np.max(path_alleles, axis=0) + 1
        for variant in range(kmers.n_variants):
            variant_kmers = []
            #n_alleles = np.max(path_alleles[:,variant])+1
            for allele in range(n_alleles[variant]):
                paths_with_allele = path_alleles[:,variant] == allele
                allele_kmers = kmers.kmers[paths_with_allele, variant]
                variant_kmers.append(allele_kmers)

            new.append(variant_kmers)

        return cls(ak.Array(new))


@dataclass
class MatrixVariantWindowKmers:
    """
    Represents kmers around variants in a 3-dimensional matrix, possible when n kmers per variant allele is fixed
    """
    kmers: Union[np.ndarray, ak.Array]  # n_paths, n variants x n_windows.
    # Can be numpy matrix if even window sizes and fixed number of alleles

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
    def from_paths_with_flexible_window_size(cls, path_sequences: PathSequences, k):
        # uses different windows for each variant, stores in an awkward array
        kmers_found = []  # one RaggedArray for each path. Each element in ragged array represents a variant allele

        for i, path in enumerate(path_sequences.iter_path_sequences()):
            variant_sizes = path[1::2].shape[1]
            window_sizes = variant_sizes + (k-1)*2
            n_kmers_in_each_window = window_sizes - k + 1
            # for each variant, find the kmers for this path and fill into the window
            starts = path._shape.starts[1::2]
            window_starts = starts - k + 1
            window_ends = window_starts + window_sizes
            windows = bnp.ragged_slice(path.ravel(), window_starts, window_ends)
            kmers = bnp.get_kmers(windows, k=k).ravel().raw().astype(np.uint64)

            kmers_found.append(
                nps.RaggedArray(kmers, n_kmers_in_each_window).tolist()
            )

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
        logging.info("Scoring all candidate kmers")
        scores = scorer.score_kmers(self.kmers.ravel()).reshape(self.kmers.shape)
        # window scores are the lowest score among all paths in windows
        #scores = np.min(scores, axis=0)
        scores = np.sum(scores, axis=0)
        # score rightmost window a bit more to prefer this on a tie
        scores[:, -1] += 1
        # best window has the highest score
        logging.info("Finding best windows based on scores")
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



def awkward_unravel_like(flat_array, like_array):
    """ Utility function for unraveling a flattened ak.Array """
