import numpy as np
import logging


class GafEntry:
    def __init__(self, read_id, nodes, mapq, score, read_length):
        self.read_id = read_id
        self.nodes = nodes
        self.mapq = mapq
        self.score = score
        self.read_length = read_length

    @classmethod
    def from_gaf_line(cls, line, edge_mapping=None):

        l = line.split()
        read_length = int(l[1])
        nodes_str = l[5]
        seperator = nodes_str[0]

        if seperator != ">" and seperator != "<":
            assert nodes_str == "*"
            score = 0
            mapq = 0
            nodes = []
        else:
            mapq = int(l[11])
            score = int(l[12].replace("AS:i:", ""))

            try:
                nodes = [int(n) for n in nodes_str.split(seperator)[1:]]
            except ValueError:
                logging.error("Failed parsing node string %s" % nodes_str)
                raise

            # add dummy nodes
            if edge_mapping is not None:
                for edge in zip(nodes[0:-1], nodes[1:]):
                    if edge in edge_mapping:
                        dummy_node = edge_mapping[edge]
                        nodes.append(dummy_node)
                        n_dummy_nodes += 1

        return cls(l[0], nodes, mapq, score, read_length)


def parse_gaf(gaf_file_name, edge_mapping=None):
    f = open(gaf_file_name)
    return (GafEntry.from_gaf_line(line, edge_mapping) for line in f)


def node_counts_from_gaf(
    gaf_file_name, edge_mapping, min_mapq=30, min_score=0, max_node_id=None
):
    n_skipped_low_mapq = 0
    n_skipped_low_score = 0
    n_dummy_nodes = 0
    all_nodes = []
    for i, mapping in enumerate(parse_gaf(gaf_file_name)):
        if i % 1000000 == 0:
            logging.info("%d reads processed" % i)

        if mapping.mapq < min_mapq:
            n_skipped_low_mapq += 1
            continue

        if mapping.score < min_score:
            n_skipped_low_score += 1
            continue

        all_nodes.extend(mapping.nodes)

    logging.info(
        "N reads skipped because too low mapq: %d (< %d)"
        % (n_skipped_low_mapq, min_mapq)
    )
    logging.info(
        "N reads skipped because too low score: %d (< %d)"
        % (n_skipped_low_score, min_score)
    )
    logging.info("%d dummy nodes were found" % n_dummy_nodes)

    node_counts = np.bincount(all_nodes, minlength=max_node_id + 1)
    return node_counts
