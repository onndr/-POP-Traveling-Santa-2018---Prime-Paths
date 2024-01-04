from typing import Callable

import numpy as np
import random

from src.constants import Point_t
from src.graphtools.graph import Graph


class Route:
    def __init__(self, points: np.array):
        self.points = points
        self.n_points = len(points)

        self.best_order = np.array(list(range(self.n_points)))  # straight order initially
        self.best_cost = self.total_distance(self.points, self.best_order)

        self.graph = None

    @staticmethod
    def distance(point1: Point_t, point2: Point_t):
        return np.sqrt((point2['X'] - point1['X']) ** 2 + (point2['Y'] - point1['Y']) ** 2)

    @staticmethod
    def total_distance(points: np.array, order: np.array):
        total = 0
        for i in range(-1, len(points)-1):
            total += Route.distance(points[order[i]], points[order[i + 1]])
        return total

    @staticmethod
    def two_opt(points: np.array, n_points: int, order: np.array, iters: int):
        best_distance = Route.total_distance(points, order)
        for _ in range(iters):
            for i in range(1, n_points - 1):
                for j in range(i + 1, n_points):
                    new_order = np.concatenate((order[:i], order[i:j + 1][::-1], order[j + 1:]))    # swap
                    new_distance = Route.total_distance(points, new_order)
                    if new_distance <= best_distance:
                        order = new_order
                        best_distance = new_distance
        return order, best_distance

    @staticmethod
    def three_opt_pick_best_variation(points, order, i, j, k):
        """ beauty is in the eye of the beholder """
        segment1 = order[:i]
        segment2 = order[i:j]
        segment3 = order[j:k]
        segment4 = order[k:]

        r1 = np.concatenate((segment1, segment2, segment3, segment4))
        r2 = np.concatenate((segment1, segment3, segment2, segment4))
        r3 = np.concatenate((segment1, segment2[::-1], segment3, segment4))
        r4 = np.concatenate((segment1, segment3[::-1], segment2, segment4))
        r5 = np.concatenate((segment1, segment2, segment4[::-1], segment3))
        r6 = np.concatenate((segment1, segment3, segment4[::-1], segment2))
        r7 = np.concatenate((segment1, segment4[::-1], segment2[::-1], segment3))
        r8 = np.concatenate((segment1, segment4[::-1], segment3[::-1], segment2))

        d1 = Route.total_distance(points, r1)
        d2 = Route.total_distance(points, r2)
        d3 = Route.total_distance(points, r3)
        d4 = Route.total_distance(points, r4)
        d5 = Route.total_distance(points, r5)
        d6 = Route.total_distance(points, r6)
        d7 = Route.total_distance(points, r7)
        d8 = Route.total_distance(points, r8)

        min_distance = min(d1, d2, d3, d4, d5, d6, d7, d8)

        if min_distance == d1:
            return r1, d1
        elif min_distance == d2:
            return r2, d2
        elif min_distance == d3:
            return r3, d3
        elif min_distance == d4:
            return r4, d4
        elif min_distance == d5:
            return r5, d5
        elif min_distance == d6:
            return r6, d6
        elif min_distance == d7:
            return r7, d7
        return r8, d8

    @staticmethod
    def three_opt(points: np.array, n_points: int, order: np.array, iters: int):
        best_distance = Route.total_distance(points, order)
        for _ in range(iters):
            for i in range(1, n_points - 3):
                for j in range(i + 2, n_points - 1):
                    for k in range(j + 2, n_points):
                        new_order, new_distance = Route.three_opt_pick_best_variation(points, order, i, j, k)
                        if new_distance < best_distance:
                            order = new_order
                            best_distance = new_distance
        return order, best_distance

    def _generate_random_solutions(self, n_solutions: int) -> list:
        solutions = []
        for _ in range(n_solutions):
            order = self.best_order.copy()
            np.random.shuffle(order)
            solutions.append(order)
        return solutions

    def _prepare_start_solutions(self, n_start_routes: int = 3, use_prev_best_solution=True):
        if use_prev_best_solution:
            init_orders = self._generate_random_solutions(max(n_start_routes, 1) - 1)
            init_orders.append(self.best_order)
        else:
            init_orders = self._generate_random_solutions(max(n_start_routes, 1))
        return init_orders

    def optimize(self, kopt_alg: Callable, iters: int = 100, n_start_routes: int = 3, use_prev_best_solution=True):
        init_orders = self._prepare_start_solutions(n_start_routes, use_prev_best_solution)
        for order in init_orders:
            order, distance = kopt_alg(self.points, self.n_points, order, iters)
            if distance <= self.best_cost:
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
