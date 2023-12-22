from queue import PriorityQueue
from typing import Callable

from src.graphtools.graph import Graph
from src.graphtools.findandunion import FAUgraph


def find_best_spanning_tree(graph: Graph, edge_evaluator: Callable):
    priority_queue = PriorityQueue()
    fau = FAUgraph()

    # put edges and their values on the queue
    for node_a in graph.edges:
        for node_b in graph.edges[node_a]:
            edge = (node_a, node_b)
            edge_value = edge_evaluator(edge)
            priority_queue.put((edge_value, edge))

    # choose best edges to create a tree
    chosen_edges = set()
    while not priority_queue.empty():
        edge_value, edge = priority_queue.get()
        success = fau.union(*edge)
        if success:
            chosen_edges.add(edge)

        if len(chosen_edges) == graph.get_node_count() - 1:
            # found enough edges to make a tree
            break
    return chosen_edges