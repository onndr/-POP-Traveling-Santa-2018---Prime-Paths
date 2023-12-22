from src.graphtools.bestspanningtree import find_best_spanning_tree
from src.graphtools.graph import Graph


def test_spanning_tree_simple():
    def edge_evaluator(edge):
        node_a, node_b = edge
        xa, ya = node_a
        xb, yb = node_b
        return xa * 2 + ya + xb * 2 + yb

    graph = Graph()
    n = 3
    for xa in range(n):
        for ya in range(n):
            for xb in range(n):
                for yb in range(n):
                    if xa == xb and ya == yb:
                        continue
                    node_a = (xa, ya)
                    node_b = (xb, yb)
                    edge = (node_a, node_b)
                    graph.add_edge(*edge)

    tree = find_best_spanning_tree(graph, edge_evaluator)
    assert tree == {((0, 0), (0, 1)),
                    ((0, 0), (0, 2)),
                    ((0, 0), (1, 0)),
                    ((0, 0), (1, 1)),
                    ((0, 0), (1, 2)),
                    ((0, 0), (2, 0)),
                    ((0, 0), (2, 1)),
                    ((0, 0), (2, 2))}
