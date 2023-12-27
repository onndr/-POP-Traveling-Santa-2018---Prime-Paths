import numpy as np
import random

from src.constants import Point_t
from src.graphtools.graph import Graph


class Route:
    def __init__(self, points: np.array):
        self.points = points
        self.n_points = len(points)

        self.straight_order = list(range(self.n_points))
        self.straight_cost = self.total_distance(self.points, self.straight_order)

        self.best_order = self.straight_order
        self.best_cost = self.straight_cost

        self.graph = None

    @staticmethod
    def distance(point1: Point_t, point2: Point_t):
        return np.sqrt((point2['X'] - point1['X']) ** 2 + (point2['Y'] - point1['Y']) ** 2)

    @staticmethod
    def total_distance(points: np.array, order: list):
        total = 0
        for i in range(-1, len(points)-1):
            total += Route.distance(points[order[i]], points[order[i + 1]])
        return total

    @staticmethod
    def _two_opt(points: np.array, n_points: int, order: list[int], iters: int):
        best_distance = Route.total_distance(points, order)
        for _ in range(iters):
            for i in range(1, n_points - 1):
                for j in range(i + 1, n_points):
                    new_order = order[:]
                    new_order[i:j] = order[j - 1:i - 1:-1]
                    new_distance = Route.total_distance(points, order)
                    if new_distance < best_distance:
                        order = new_order
                        best_distance = new_distance
        return order, best_distance

    def _generate_random_solutions(self, n_solutions: int) -> list:
        solutions = []
        for _ in range(n_solutions):
            order = self.straight_order.copy()
            random.shuffle(order)
            solutions.append(order)
        return solutions

    def two_opt(self, iters: int = 100, n_start_routes: int = 3):
        init_orders = self._generate_random_solutions(n_start_routes)
        for order in init_orders:
            order, distance = self._two_opt(self.points, self.n_points, order, iters)
            if distance < self.best_cost:
                self.best_order = order
                self.best_cost = distance
                self.graph = None   # lazy graph, only recreated on request
        return self.best_order, self.best_cost

    def get_graph(self) -> Graph:
        if not self.graph:
            self.graph = Graph()
            for i in range(-1, self.n_points-1):
                self.graph.add_edge(self.points[self.best_order[i]], self.points[self.best_order[i+1]])

        return self.graph
