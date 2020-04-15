from alignment_free_graph_genotyper.chain_genotyper import ChainGenotyper
import numpy as np

def test_simple():
    read_offsets = [10, 20, 50, 51, 100, 100]
    ref_offsets = [11, 21, 99, 100, 101, 200]
    nodes = [1, 2, 50, 52, 4, 100]

    chains = ChainGenotyper.find_chains(np.array(ref_offsets), np.array(read_offsets), np.array(nodes))
    print(chains)

    assert list(chains[0][1]) == [1, 2, 4]
    assert list(chains[1][1]) == [50, 52]
    assert list(chains[2][1]) == [100]

if __name__ == "__main__":
    test_simple()