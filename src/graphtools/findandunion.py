class FAUgraph:
    def __init__(self):
        self.assignments = dict()

    def union(self, node_a, node_b):
        parent_a = self.find_parent(node_a)
        parent_b = self.find_parent(node_b)

        if parent_a == parent_b:
            return False

        new_parent = min(parent_a, parent_b)
        self.assignments[parent_a] = new_parent
        self.assignments[parent_b] = new_parent
        return True

    def find_parent(self, node):
        if node not in self.assignments:
            self.assignments[node] = node
            return node

        parent = self.assignments[node]
        if parent != node:
            parent = self.find_parent(parent)
            self.assignments[node] = parent
        return parent

    def clear(self):
        self.assignments = dict()
