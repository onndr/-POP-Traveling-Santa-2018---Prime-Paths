class Graph:
    def __init__(self):
        self.edges = dict()

    def get_node_count(self):
        return len(self.edges.keys())

    def remove_edge(self, node_a, node_b):
        self.edges[node_a].remove(node_b)
        self.edges[node_b].remove(node_a)

    def add_edge(self, node_a, node_b):
        # bi-directional graph
        if node_a not in self.edges:
            self.edges[node_a] = set()
        self.edges[node_a].add(node_b)

        if node_b not in self.edges:
            self.edges[node_b] = set()
        self.edges[node_b].add(node_a)

    def get_edge_data(self):
        return self.edges

    def get_node_edges(self, node):
        return self.edges[node]
