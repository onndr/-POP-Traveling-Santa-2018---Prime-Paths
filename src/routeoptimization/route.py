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
    def total_distance(points: np.array, order: np.array, with_primes_penalty: bool = False):
        total = 0
        for i in range(-1, len(points) - 1):
            if with_primes_penalty and i % 10 == 0 and not points[order[i]]['prime']:
                total += Route.distance(points[order[i]], points[order[i + 1]]) * 1.1
            else:
                total += Route.distance(points[order[i]], points[order[i + 1]])
        return total

    @staticmethod
    def two_opt(points: np.array, n_points: int, order: np.array, iters: int):
        best_distance = Route.total_distance(points, order)

        def gain_from_2_opt_swap(x1, x2, y1, y2):
            old_len = Route.distance(points[x1], points[x2]) + Route.distance(points[y1], points[y2])
            new_len = Route.distance(points[x1], points[y1]) + Route.distance(points[x2], points[y2])
            return old_len - new_len

        for _ in range(iters):
            i = random.randint(0, n_points - 3)
            x1 = order[i]
            x2 = order[i + 1 % n_points]

            if i == 0:
                j_limit = n_points - 2
            else:
                j_limit = n_points - 1

            j = random.randint(i + 2, j_limit)
            y1 = order[j]
            y2 = order[(j + 1) % n_points]

            if gain_from_2_opt_swap(x1, x2, y1, y2) > 0:
                new_order = np.concatenate((order[:i + 1], order[i + 1:j + 1][::-1], order[j + 1:]))
                new_distance = Route.total_distance(points, new_order)
                if new_distance <= best_distance:
                    order = new_order
                    best_distance = new_distance

        return order, best_distance

    @staticmethod
    def three_opt(points: np.array, n_points: int, order: np.array, iters: int):
        def gain_from_3_opt_swap(x1, x2, y1, y2, z1, z2, opt_recombination):
            old_len = 0
            new_len = 0
            match opt_recombination:
                case 0:
                    return 0
                case 1:
                    old_len = Route.distance(points[x1], points[x2]) + Route.distance(points[z1], points[z2])
                    new_len = Route.distance(points[x1], points[z1]) + Route.distance(points[x2], points[z2])
                case 2:
                    old_len = Route.distance(points[y1], points[y2]) + Route.distance(points[z1], points[z2])
                    new_len = Route.distance(points[y1], points[z1]) + Route.distance(points[y2], points[z2])
                case 3:
                    old_len = Route.distance(points[x1], points[x2]) + Route.distance(points[y1], points[y2])
                    new_len = Route.distance(points[x1], points[y1]) + Route.distance(points[x2], points[y2])
                case 4:
                    new_len = (Route.distance(points[x1], points[y1]) +
                               Route.distance(points[x2], points[z1]) +
                               Route.distance(points[y2], points[z2]))
                case 5:
                    new_len = (Route.distance(points[x1], points[z1]) +
                               Route.distance(points[y2], points[x2]) +
                               Route.distance(points[y1], points[z2]))
                case 6:
                    new_len = (Route.distance(points[x1], points[y2]) +
                               Route.distance(points[z1], points[y1]) +
                               Route.distance(points[x2], points[z2]))
                case 7:
                    new_len = (Route.distance(points[x1], points[y2]) +
                               Route.distance(points[z1], points[x2]) +
                               Route.distance(points[y1], points[z2]))
                case _:
                    raise ValueError("Invalid opt_recombination value")

            if opt_recombination in [4, 5, 6, 7]:
                old_len = (Route.distance(points[x1], points[x2]) +
                           Route.distance(points[y1], points[y2]) +
                           Route.distance(points[z1], points[z2]))

            return old_len - new_len

        def three_opt_recombine(order, i, j, k, opt_recombination):
            segment1 = np.concatenate((order[k + 1 % n_points:], order[:i + 1 % n_points]))
            segment2 = order[i + 1 % n_points: j + 1 % n_points]
            segment3 = order[j + 1 % n_points: k + 1 % n_points]

            match opt_recombination:
                case 0:
                    return order
                case 1:
                    segment1 = segment1[::-1]
                case 2:
                    segment3 = segment3[::-1]
                case 3:
                    segment2 = segment2[::-1]
                case 4:
                    segment3 = segment3[::-1]
                    segment2 = segment2[::-1]
                case 5:
                    segment1 = segment1[::-1]
                    segment2 = segment2[::-1]
                case 6:
                    segment1 = segment1[::-1]
                    segment3 = segment3[::-1]
                case 7:
                    segment1 = segment1[::-1]
                    segment2 = segment2[::-1]
                    segment3 = segment3[::-1]
                case _:
                    raise ValueError("Invalid opt_recombination value")

            return np.concatenate((segment1, segment2, segment3))

        best_distance = Route.total_distance(points, order)
        for _ in range(iters):
            i = random.randint(0, n_points - 1)
            x1 = order[i]
            x2 = order[(i + 1) % n_points]

            j = random.randint(i + 1, i + n_points - 3) % n_points
            y1 = order[j]
            y2 = order[(j + 1) % n_points]

            k = random.randint(i + 2, i + n_points - 1) % n_points
            z1 = order[k]
            z2 = order[(k + 1) % n_points]

            i, j, k = sorted([i, j, k])

            for opt_recombination in range(1, 8):
                if gain_from_3_opt_swap(x1, x2, y1, y2, z1, z2, opt_recombination) > 0:
                    new_order = three_opt_recombine(order, i, j, k, opt_recombination)
                    new_distance = Route.total_distance(points, new_order)
                    if new_distance <= best_distance:
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

    def _prepare_start_solutions(self, n_start_routes: int = 3, use_prev_best_solution=True,
                                 copy_prev_best_solution=True):
        if use_prev_best_solution:
            if copy_prev_best_solution:
                return [self.best_order.copy() for _ in range(n_start_routes)]
            else:
                init_orders = self._generate_random_solutions(n_start_routes - 1)
                init_orders.append(self.best_order)
                return init_orders
        return self._generate_random_solutions(n_start_routes)

    def optimize(self, kopt_alg: Callable, iters: int = 100, n_start_routes: int = 3, use_prev_best_solution=True,
                 copy_prev_best_solution=True):
        init_orders = self._prepare_start_solutions(n_start_routes, use_prev_best_solution, copy_prev_best_solution)
        for order in init_orders:
            order, distance = kopt_alg(self.points, self.n_points, order, iters)
            if distance <= self.best_cost:
                self.best_order = order
                self.best_cost = distance
                self.graph = None  # lazy graph, only recreated on request
        return self.best_order, self.best_cost

    def get_graph(self) -> Graph:
        if not self.graph:
            self.graph = Graph()
            for i in range(-1, self.n_points - 1):
                self.graph.add_edge(self.points[self.best_order[i]], self.points[self.best_order[i + 1]])

        return self.graph
