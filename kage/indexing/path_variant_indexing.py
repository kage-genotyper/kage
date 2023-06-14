import itertools
import logging
import time
from dataclasses import dataclass
import npstructures as nps
import numpy as np
import bionumpy as bnp
from graph_kmer_index import KmerIndex, FlatKmers
import tqdm
from kage.indexing.graph import Graph

from .paths import Paths
from .sparse_haplotype_matrix import SparseHaplotypeMatrix
from ..models.mapping_model import LimitedFrequencySamplingComboModel
from .tricky_variants import TrickyVariants

from ..preprocessing.variants import Variants
from ..util import log_memory_usage_now

"""
Module for simple variant signature finding by using static predetermined paths through the "graph".

Works when all variants are biallelic and there are no overlapping variants.
"""


@dataclass
class MatrixVariantWindowKmers:
    """
    Represents kmers around variants in a 3-dimensional matrix, possible when n kmers per variant allele is fixed
    """
    kmers: np.ndarray  # n_paths, n variants x n_windows
    variant_alleles: np.ndarray

    @classmethod
    def from_paths(cls, paths, k):
        n_variants = paths.n_variants()
        n_paths = len(paths.paths)
        n_windows = k - 1 - k//2
        matrix = np.zeros((n_paths, n_variants, n_windows), dtype=np.uint64)

        for i, path in enumerate(paths.iter_path_sequences()):
            starts = path._shape.starts[1::2]
            window_starts = starts - k + 1 + k//2
            #window_ends = np.minimum(starts + k, path.size)
            window_ends = starts + k - 1
            windows = bnp.ragged_slice(path.ravel(), window_starts, window_ends)
            kmers = bnp.get_kmers(windows, k=k).ravel().raw().astype(np.uint64)
            kmers = kmers.reshape((n_variants, n_windows))
            matrix[i, :, :] = kmers

        return cls(matrix, paths.variant_alleles)

    def get_kmers(self, variant, allele):
        relevant_paths = np.where(self.variant_alleles[:, variant] == allele)[0]
        return self.kmers[relevant_paths, variant, :]

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


class FastApproxCounter:
    """ Fast counter that uses modulo and allows collisions"""
    def __init__(self, array, modulo):
        self._array = array
        self._modulo = modulo

    @classmethod
    def empty(cls, modulo):
        return cls(np.zeros(modulo, dtype=np.int16), modulo)

    def add(self, values):
        self._array[values % self._modulo] += 1

    @classmethod
    def from_keys_and_values(cls, keys, values, modulo):
        array = np.zeros(modulo, dtype=np.int16)
        array[keys % modulo] = values
        return cls(array, modulo)

    def __getitem__(self, keys):
        return self._array[keys % self._modulo]

    def score_kmers(self, kmers):
        return -self[kmers]



def make_kmer_scorer_from_random_haplotypes(graph: Graph, haplotype_matrix: SparseHaplotypeMatrix,
                                            k: int,
                                            n_haplotypes: int = 4,
                                            modulo: int = 20000033):
    """
    Estimates counts from random individuals
    """
    counter = FastApproxCounter.empty(modulo)
    chosen_haplotypes = np.random.choice(np.arange(haplotype_matrix.n_haplotypes), n_haplotypes, replace=False)
    logging.info("Picked random haplotypes to make kmer scorer: %s" % chosen_haplotypes)
    haplotype_nodes = (haplotype_matrix.get_haplotype(haplotype) for haplotype in chosen_haplotypes)

    # also add the reference and a haplotype with all variants
    haplotype_nodes = itertools.chain(haplotype_nodes,
                                      [np.zeros(haplotype_matrix.n_variants, dtype=np.uint8),
                                        np.ones(haplotype_matrix.n_variants, dtype=np.uint8)])

    for i, nodes in tqdm.tqdm(enumerate(haplotype_nodes), desc="Estimating global kmer counts", total=len(chosen_haplotypes), unit="haplotype"):
        #haplotype_nodes = haplotype_matrix.get_haplotype(haplotype)
        log_memory_usage_now("Memory after getting nodes")
        kmers = graph.get_haplotype_kmers(nodes, k=k, stream=True)
        log_memory_usage_now("Memory after kmers")
        for subkmers in kmers:
            counter.add(subkmers)
        log_memory_usage_now("After adding haplotype %d" % i)

    # also add the reference and a haplotype with all variants
    #kmers = graph.get_haplotype_kmers(np.zeros(haplotype_matrix.n_variants, dtype=np.uint8), k=k, stream=False)
    #print(kmers)
    #counter.add(kmers)
    #counter.add(graph.get_haplotype_kmers(np.ones(haplotype_matrix.n_variants, dtype=np.uint8), k=k))

    return counter


class Scorer:
    def __init__(self):
        pass

    def score_kmers(self, kmers):
        pass


class KmerFrequencyScorer(Scorer):
    def __init__(self, frequency_lookup: nps.HashTable):
        self._frequency_lookup = frequency_lookup

    def score_kmers(self, kmers):
        return np.max(self._frequency_lookup[kmers])



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

    def run(self) -> Signatures:
        """
        Returns a list of kmers. List contains alleles, each RaggedArray represents the variants
        """
        raise NotImplementedError("Not used anymore")
        chosen_ref_kmers = []
        chosen_alt_kmers = []

        for variant in tqdm.tqdm(range(self._paths.n_variants()), desc="Finding signatures", unit="variants", total=self._paths.n_variants()):
            best_kmers = None
            best_score = -10000000000000000000
            all_ref_kmers = self._paths.get_kmers(variant, 0, self._k)
            all_alt_kmers = self._paths.get_kmers(variant, 1, self._k)

            assert np.all(all_ref_kmers.shape[1] == all_alt_kmers.shape[1])

            for window in range(len(all_ref_kmers[0])):
                ref_kmers = np.unique(all_ref_kmers[:, window]) #.raw().ravel())
                alt_kmers = np.unique(all_alt_kmers[:, window]) #.raw().ravel())

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


class SignatureFinder2(SignatureFinder):
    def run(self) -> Signatures:
        # for each path, first creates a RaggedArray with windows for each variant
        paths = MatrixVariantWindowKmers.from_paths(self._paths, self._k)  # ._paths.get_windows_around_variants(self._k)
        chosen_ref_kmers = []
        chosen_alt_kmers = []

        for variant in tqdm.tqdm(range(self._paths.n_variants()), desc="Finding signatures", unit="variants",
                                 total=self._paths.n_variants()):
            best_kmers = None
            best_score = -10000000000000000000
            all_ref_kmers = paths.get_kmers(variant, 0)
            all_alt_kmers = paths.get_kmers(variant, 1)
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
        kmers = MatrixVariantWindowKmers.from_paths(self._paths, self._k)
        self._all_possible_kmers = kmers
        best_kmers = kmers.get_best_kmers(self._scorer)

        # hack to split matrix into ref and alt kmers
        n_variants = self._paths.n_variants()
        n_paths = len(self._paths.paths)
        all_ref_kmers = best_kmers.T[kmers.variant_alleles.T == 0].reshape((n_variants, n_paths//2))
        all_alt_kmers = best_kmers.T[kmers.variant_alleles.T == 1].reshape((n_variants, n_paths//2))


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

        if variants is not None:
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


            """
            # improvement would be to allow different window pos at ref and alt
            # todo: Find all okay windows, and pick the one with the highest score
            if len(set(ref_kmers).intersection(unique_alt)) == 0 and len(set(alt_kmers).intersection(unique_ref)) == 0:
                # this window is good
                self._chosen_ref_kmers[id] = ref_kmers
                self._chosen_alt_kmers[id] = alt_kmers
                n_alt_replaced += 1
                n_ref_replaced += 1
                print(f"REPLACED INDEL {id}")
                print(f"Picked window {window}")
                print("New kmers ref", ref_kmers)
                print("New kmers alt", alt_kmers)
                break
            """


            """
            unique_options_ref = np.setdiff1d(options_ref, options_alt)
            unique_options_alt = np.setdiff1d(options_alt, options_ref)

            if len(unique_options_ref) > 0 and len(unique_options_alt) > 0:
                score_ref = self._scorer.score_kmers(unique_options_ref)
                #print("Best score ref: ", np.max(score_ref))
                best_ref = unique_options_ref[np.argmax(score_ref)]
                n_ref_replaced += 1

                score_alt = self._scorer.score_kmers(unique_options_alt)
                best_alt = unique_options_alt[np.argmax(score_alt)]
                n_alt_replaced += 1
                print("REPLACED INDEL {id}")
                print("Old ref", options_ref)
                print("Old alt", options_alt)
                print("New ref", best_ref)
                print("New alt", best_alt)
                print("Picked in old", self._chosen_ref_kmers[id], self._chosen_alt_kmers[id])
                print()
                self._chosen_ref_kmers[id] = np.array([best_ref])
                self._chosen_alt_kmers[id] = np.array([best_alt])
            """

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



class SignaturesWithNodes:
    """
    Represents signatures compatible with graph-nodes.
    Nodes are given implicitly from variant ids. Makes it possible
    to create a kmer index that requires nodes.
    """
    pass


class MappingModelCreator:
    def __init__(self, graph: Graph, kmer_index: KmerIndex,
                 haplotype_matrix: SparseHaplotypeMatrix,
                 max_count=10, k=31):
        self._graph = graph
        self._kmer_index = kmer_index
        self._haplotype_matrix = haplotype_matrix
        self._n_nodes = graph.n_nodes()
        self._counts = LimitedFrequencySamplingComboModel.create_empty(self._n_nodes, max_count)
        logging.info("Inited empty model")
        self._k = k
        self._max_count = max_count

    def _process_individual(self, i):
        logging.info("Processing individual %d" % i)
        t0 = time.perf_counter()
        # extract kmers from both haplotypes and map these using the kmer index
        haplotype1 = self._haplotype_matrix.get_haplotype(i*2)
        haplotype2 = self._haplotype_matrix.get_haplotype(i*2+1)
        logging.info("Got haplotypes, took %.2f seconds" % (time.perf_counter() - t0))

        haplotype1_nodes = self._haplotype_matrix.get_haplotype_nodes(i*2)
        haplotype2_nodes = self._haplotype_matrix.get_haplotype_nodes(i*2+1)


        sequence1 = self._graph.sequence(haplotype1).ravel()
        sequence2 = self._graph.sequence(haplotype2).ravel()
        logging.info("Got haplotypes, total now: %.2f seconds" % (time.perf_counter() - t0))

        kmers1 = bnp.get_kmers(sequence1, self._k).ravel().raw().astype(np.uint64)
        kmers2 = bnp.get_kmers(sequence2, self._k).ravel().raw().astype(np.uint64)
        logging.info("Got kmers, total now: %.2f seconds" % (time.perf_counter() - t0))

        t0_mapping_kmers = time.perf_counter()
        node_counts = self._kmer_index.map_kmers(kmers1, self._n_nodes)
        node_counts += self._kmer_index.map_kmers(kmers2, self._n_nodes)
        logging.info("Mapped kmers, took: %.2f seconds" % (time.perf_counter() - t0_mapping_kmers))

        self._add_node_counts(haplotype1_nodes, haplotype2_nodes, node_counts)
        logging.info("Done, total now: %.2f seconds" % (time.perf_counter() - t0))

    def _add_node_counts(self, haplotype1_nodes, haplotype2_nodes, node_counts):
        # split into nodes that the haplotype has and nodes not
        # mask represents the number of haplotypes this individual has per node (0, 1 or 2 for diploid individuals)
        mask = np.zeros(self._n_nodes, dtype=np.int8)
        mask[haplotype1_nodes] += 1
        mask[haplotype2_nodes] += 1
        for genotype in [0, 1, 2]:
            nodes_with_genotype = np.where(mask == genotype)[0]
            counts_on_nodes = node_counts[nodes_with_genotype].astype(int)
            below_max_count = np.where(counts_on_nodes < self._max_count)[
                0]  # ignoring counts larger than supported by matrix
            self._counts.diplotype_counts[genotype][
                nodes_with_genotype[below_max_count], counts_on_nodes[below_max_count]
            ] += 1

    def run(self) -> LimitedFrequencySamplingComboModel:
        n_variants, n_haplotypes = self._haplotype_matrix.shape
        n_nodes = n_variants * 2
        for individual in tqdm.tqdm(range(n_haplotypes // 2), total=n_haplotypes // 2, desc="Creating count model", unit="individuals"):
            self._process_individual(individual)

        return self._counts




def make_linear_reference_kmer_counter(reference_file_name, k=31, modulo=20000033):
    """
    Gets all kmers from a linear reference.
    """
    from graph_kmer_index.kmer_counter import KmerCounter
    logging.info("Getting kmers from linear ref")
    reference = bnp.open(reference_file_name).read()
    reference.sequence[reference.sequence == "N"] = "A"  # tmp hack to allow kmer encoding

    kmers = bnp.get_kmers(reference.sequence, k)
    kmers = kmers.raw().ravel().astype(np.uint64)
    
    from graph_kmer_index.kmer_hashing import kmer_hashes_to_reverse_complement_hash
    logging.info("Getting reverse complements")
    revcomp = kmer_hashes_to_reverse_complement_hash(kmers, k)
    kmers = np.concatenate([kmers, revcomp])

    counter = KmerCounter.from_kmers(kmers, modulo)
    return counter



def make_scorer_from_paths(paths: Paths, k, modulo):
    kmers = []
    for path in tqdm.tqdm(paths.paths, desc="Getting kmers from paths", unit="paths"):
        kmers.append(bnp.get_kmers(path.ravel(), k=k).raw().ravel().astype(np.uint64))
        kmers.append(bnp.get_kmers(bnp.sequence.get_reverse_complement(path).ravel(), k=k).raw().ravel().astype(np.uint64))

    kmers = np.concatenate(kmers).astype(np.int64)

    # using fastapproccounter, we only need to bincount
    counter = np.bincount(kmers % modulo, minlength=modulo)
    return FastApproxCounter(counter, modulo=modulo)

    #logging.info("Getting unique kmers and counts")
    #unique, counts = np.unique(kmers, return_counts=True)
    #return FastApproxCounter.from_keys_and_values(unique, counts, modulo=modulo)


def find_tricky_variants_from_signatures(signatures: Signatures):
    """
    Finds variants with bad signatures.
    """
    nonunique_ref = np.in1d(signatures.ref.ravel(), signatures.alt.ravel())
    nonunique_alt = np.in1d(signatures.alt.ravel(), signatures.ref.ravel())
    #mask_ref = np.zeros_like(signatures.ref, dtype=bool)
    #mask_ref[nonunique_ref] = True
    mask_ref = nps.RaggedArray(nonunique_ref, signatures.ref.shape)
    mask_alt = nps.RaggedArray(nonunique_alt, signatures.alt.shape)
    print(nonunique_ref, nonunique_alt)
    #mask_alt = np.zeros_like(signatures.alt, dtype=bool)
    #mask_alt[nonunique_alt] = True

    tricky_variants_ref = np.any(mask_ref, axis=1)
    tricky_variants_alt = np.any(mask_alt, axis=1)
    tricky_variants = np.logical_or(tricky_variants_ref, tricky_variants_alt)

    logging.info(f"{np.sum(tricky_variants)} tricky variants")
    return TrickyVariants(tricky_variants)


def find_tricky_variants_from_signatures2(signatures: Signatures):
    n_tricky_no_signatures = 0
    tricky_variants = np.zeros(signatures.ref.shape[0], dtype=bool)
    for i, (ref, alt) in tqdm.tqdm(enumerate(zip(signatures.ref, signatures.alt)), total=signatures.ref.shape[0], desc="Finding tricky variants", unit="variants"):
        if len(ref) == 0 or len(alt) == 0:
            tricky_variants[i] = True
            n_tricky_no_signatures += 1
        if set(ref).intersection(alt):
            tricky_variants[i] = True

    logging.info(f"{np.sum(tricky_variants)} tricky variants")
    logging.info(f"Tricky because no signatures: {n_tricky_no_signatures}")
    return TrickyVariants(tricky_variants)


def find_tricky_variants_with_count_model(signatures: Signatures, model):
    tricky = find_tricky_variants_from_signatures2(signatures).tricky_variants

    # also check if model is missing data
    n_missing_model = 0
    for variant in tqdm.tqdm(range(len(signatures.ref.shape[1])), total=signatures.ref.shape[0], desc="Finding tricky variants", unit="variants"):
        ref_node = variant*2
        alt_node = variant*2 + 1
        if model.has_no_data(ref_node, threshold=3) or model.has_no_data(alt_node, threshold=3):
            tricky[variant] = True
            n_missing_model += 1
            print(model.describe_node(ref_node))
            print(model.describe_node(alt_node))

    logging.info("N tricky variants due to missing data in model: %d" % n_missing_model)
    return TrickyVariants(tricky)



