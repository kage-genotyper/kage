from graph_kmer_index import SnpKmerFinder, KmerIndex
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval, Block, Interval
from alignment_free_graph_genotyper.genotyper import IndependentKmerGenotyper, ReadKmers, BestChainGenotyper
import numpy as np


def test_simple():
    graph = Graph({1: Block(10), 2: Block(1), 3: Block(1), 4: Block(10)}, {1: [2, 3], 2: [4], 3: [4]})
    graph.convert_to_numpy_backend()

    sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
    sequence_graph.set_sequence(1, "GGGTTTATAC")
    sequence_graph.set_sequence(2, "A")
    sequence_graph.set_sequence(3, "C")
    sequence_graph.set_sequence(4, "GTACAGTGTA")

    linear_ref = Interval(0, 10, [1, 3, 4], graph)
    linear_ref = linear_ref.to_numpy_indexed_interval()

    finder = SnpKmerFinder(graph, sequence_graph, linear_ref, k=3)
    flat_index = finder.find_kmers()

    kmer_index = KmerIndex.from_multiple_flat_kmers([flat_index])


    read_kmers = ReadKmers.from_list_of_string_kmers([("GGG", "GGT", "GTT"),
                  ("ACA", "CAG", "AGT")])

    print(list(read_kmers))

    vcf_lines = [
        "1  11  .   C   A   .   PASS    AF=0.5;AC=20\n"
    ]
    with open("tmp.vcf", "w") as f:
        f.writelines(vcf_lines)


    genotyper = IndependentKmerGenotyper(graph, sequence_graph, linear_ref, read_kmers, kmer_index, "tmp.vcf", k=3)
    genotyper.genotype()


def test_simple_chaining():
    nodes = np.array([1, 2, 3, 4, 10, 11, 12, 13, 14])
    offsets = np.array([4, 4, 5, 6, 10, 10, 10, 11, 11])
    assert list(BestChainGenotyper.get_nodes_in_best_chain(nodes, offsets, expected_read_length=3)) == [1, 2, 3, 4]

    nodes = np.array([1, 2, 3, 4, 10, 11, 12, 14, 13])
    offsets = np.array([4, 4, 4, 6, 10, 10, 10, 12, 11])
    assert list(BestChainGenotyper.get_nodes_in_best_chain(nodes, offsets, expected_read_length=3)) == [10, 11, 12, 13, 14]

    nodes = np.array([1, 2, 3, 4, 10, 11, 12, 14, 13])
    offsets = np.array([4, 4, 4, 6, 10, 10, 10, 12, 11])
    assert list(BestChainGenotyper.get_nodes_in_best_chain(nodes, offsets, expected_read_length=100)) == [1, 2, 3, 4, 10, 11, 12, 13, 14]

if __name__ == "__main__":
    #test_simple()
    test_simple_chaining()
