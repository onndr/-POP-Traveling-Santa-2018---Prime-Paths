from src.graphtools.graph import Graph


def test_graph_simple():
    graph = Graph()
    p1 = 1
    p2 = 2
    p3 = 3
    p4 = 4
    p5 = 5
    p6 = 6
    graph.add_edge(p1, p2)
    graph.add_edge(p1, p3)
    graph.add_edge(p2, p4)
    graph.add_edge(p2, p4)
    assert graph.get_node_count() == 4
    assert graph.get_node_edges(p1) == {p2, p3}
    assert graph.get_node_edges(p2) == {p1, p4}

    graph.remove_edge(p4, p2)
    assert graph.get_node_count() == 4
    assert graph.get_node_edges(p2) == {p1}
    assert len(graph.get_node_edges(p4)) == 0
    assert graph.get_node_edges(p1) == {p2, p3}
