from typing import Callable

import numpy as np
import random

from src.graphtools.graph import Graph


class Route:
    def __init__(self, points: np.array):
        self.points = points
        self.n_points = len(points)

        self.best_order = np.array(list(range(self.n_points)))  # straight order initially
        self.best_cost = self.total_distance(self.points, self.best_order)

        self.graph = None

    @staticmethod
    def total_distance(points: np.array, order: np.array, with_primes_penalty: bool = False):
        total = 0
        for i in range(-1, len(order) - 1):
            point_ = points[order[i]]
            point_1 = points[order[i + 1]]
            subtotal = np.sqrt((point_1['X'] - point_['X']) ** 2 + (point_1['Y'] - point_['Y']) ** 2)
            total += subtotal * 1.1 if with_primes_penalty and i % 10 == 0 and not points[order[i]]['prime'] else subtotal
        return total

    @staticmethod
    def two_opt(points: np.array, n_points: int, order: np.array, iters: int):
        best_distance = Route.total_distance(points, order)

        def gain_from_2_opt_swap(x1, x2, y1, y2):
            point_ = points[x1]
            point_1 = points[x2]
            point_2 = points[y1]
            point_3 = points[y2]
            old_len = np.sqrt((point_1['X'] - point_['X']) ** 2 + (point_1['Y'] - point_['Y']) ** 2) + np.sqrt(
                (point_3['X'] - point_2['X']) ** 2 + (
                        point_3['Y'] - point_2['Y']) ** 2)
            point_4 = points[x1]
            point_5 = points[y1]
            point_6 = points[x2]
            point_7 = points[y2]
            new_len = np.sqrt((point_5['X'] - point_4['X']) ** 2 + (point_5['Y'] - point_4['Y']) ** 2) + np.sqrt((
                                                                                                                         point_7[
                                                                                                                             'X'] -
                                                                                                                         point_6[
                                                                                                                             'X']) ** 2 + (
                                                                                                                         point_7[
                                                                                                                             'Y'] -
                                                                                                                         point_6[
                                                                                                                             'Y']) ** 2)
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
    def two_opt_stat(points: np.array, n_points: int, order: np.array, iters: int):
        best_distance = Route.total_distance(points, order)

        def gain_from_2_opt_swap(x1, x2, y1, y2):
            point_ = points[x1]
            point_1 = points[x2]
            point_2 = points[y1]
            point_3 = points[y2]
            old_len = np.sqrt((point_1['X'] - point_['X']) ** 2 + (point_1['Y'] - point_['Y']) ** 2) + np.sqrt(
                (point_3['X'] - point_2['X']) ** 2 + (
                        point_3['Y'] - point_2['Y']) ** 2)
            point_4 = points[x1]
            point_5 = points[y1]
            point_6 = points[x2]
            point_7 = points[y2]
            new_len = np.sqrt((point_5['X'] - point_4['X']) ** 2 + (point_5['Y'] - point_4['Y']) ** 2) + np.sqrt((
                                                                                                                         point_7[
                                                                                                                             'X'] -
                                                                                                                         point_6[
                                                                                                                             'X']) ** 2 + (
                                                                                                                         point_7[
                                                                                                                             'Y'] -
                                                                                                                         point_6[
                                                                                                                             'Y']) ** 2)
            return old_len - new_len

        intermediate_results = []
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

            intermediate_results.append(best_distance)

        return order, best_distance, intermediate_results

    @staticmethod
    def three_opt(points: np.array, n_points: int, order: np.array, iters: int):
        def gain_from_3_opt_swap(x1, x2, y1, y2, z1, z2, opt_recombination):
            old_len = 0
            new_len = 0
            match opt_recombination:
                case 0:
                    return 0
                case 1:
                    point_ = points[x1]
                    point_1 = points[x2]
                    point_2 = points[z1]
                    point_3 = points[z2]
                    old_len = np.sqrt((point_1['X'] - point_['X']) ** 2 + (point_1['Y'] - point_['Y']) ** 2) + np.sqrt((
                                                                                                                               point_3[
                                                                                                                                   'X'] -
                                                                                                                               point_2[
                                                                                                                                   'X']) ** 2 + (
                                                                                                                               point_3[
                                                                                                                                   'Y'] -
                                                                                                                               point_2[
                                                                                                                                   'Y']) ** 2)
                    point_4 = points[x1]
                    point_5 = points[z1]
                    point_6 = points[x2]
                    point_7 = points[z2]
                    new_len = np.sqrt((point_5['X'] - point_4['X']) ** 2 + (
                            point_5['Y'] - point_4['Y']) ** 2) + np.sqrt((point_7['X'] - point_6['X']) ** 2 + (
                            point_7['Y'] - point_6['Y']) ** 2)
                case 2:
                    point_8 = points[y1]
                    point_9 = points[y2]
                    point_10 = points[z1]
                    point_11 = points[z2]
                    old_len = np.sqrt((point_9['X'] - point_8['X']) ** 2 + (
                            point_9['Y'] - point_8['Y']) ** 2) + np.sqrt((point_11['X'] - point_10['X']) ** 2 + (
                            point_11['Y'] - point_10['Y']) ** 2)
                    point_12 = points[y1]
                    point_13 = points[z1]
                    point_14 = points[y2]
                    point_15 = points[z2]
                    new_len = np.sqrt((point_13['X'] - point_12['X']) ** 2 + (
                            point_13['Y'] - point_12['Y']) ** 2) + np.sqrt((point_15['X'] - point_14['X']) ** 2 + (
                            point_15['Y'] - point_14['Y']) ** 2)
                case 3:
                    point_16 = points[x1]
                    point_17 = points[x2]
                    point_18 = points[y1]
                    point_19 = points[y2]
                    old_len = np.sqrt((point_17['X'] - point_16['X']) ** 2 + (
                            point_17['Y'] - point_16['Y']) ** 2) + np.sqrt((point_19['X'] - point_18['X']) ** 2 + (
                            point_19['Y'] - point_18['Y']) ** 2)
                    point_20 = points[x1]
                    point_21 = points[y1]
                    point_22 = points[x2]
                    point_23 = points[y2]
                    new_len = np.sqrt((point_21['X'] - point_20['X']) ** 2 + (
                            point_21['Y'] - point_20['Y']) ** 2) + np.sqrt((point_23['X'] - point_22['X']) ** 2 + (
                            point_23['Y'] - point_22['Y']) ** 2)
                case 4:
                    point_24 = points[x1]
                    point_25 = points[y1]
                    point_26 = points[x2]
                    point_27 = points[z1]
                    point_28 = points[y2]
                    point_29 = points[z2]
                    new_len = (np.sqrt((point_25['X'] - point_24['X']) ** 2 + (point_25['Y'] - point_24['Y']) ** 2) +
                               np.sqrt((point_27['X'] - point_26['X']) ** 2 + (point_27['Y'] - point_26['Y']) ** 2) +
                               np.sqrt((point_29['X'] - point_28['X']) ** 2 + (point_29['Y'] - point_28['Y']) ** 2))
                case 5:
                    point_30 = points[x1]
                    point_31 = points[z1]
                    point_32 = points[y2]
                    point_33 = points[x2]
                    point_34 = points[y1]
                    point_35 = points[z2]
                    new_len = (np.sqrt((point_31['X'] - point_30['X']) ** 2 + (point_31['Y'] - point_30['Y']) ** 2) +
                               np.sqrt((point_33['X'] - point_32['X']) ** 2 + (point_33['Y'] - point_32['Y']) ** 2) +
                               np.sqrt((point_35['X'] - point_34['X']) ** 2 + (point_35['Y'] - point_34['Y']) ** 2))
                case 6:
                    point_36 = points[x1]
                    point_37 = points[y2]
                    point_38 = points[z1]
                    point_39 = points[y1]
                    point_40 = points[x2]
                    point_41 = points[z2]
                    new_len = (np.sqrt((point_37['X'] - point_36['X']) ** 2 + (point_37['Y'] - point_36['Y']) ** 2) +
                               np.sqrt((point_39['X'] - point_38['X']) ** 2 + (point_39['Y'] - point_38['Y']) ** 2) +
                               np.sqrt((point_41['X'] - point_40['X']) ** 2 + (point_41['Y'] - point_40['Y']) ** 2))
                case 7:
                    point_42 = points[x1]
                    point_43 = points[y2]
                    point_44 = points[z1]
                    point_45 = points[x2]
                    point_46 = points[y1]
                    point_47 = points[z2]
                    new_len = (np.sqrt((point_43['X'] - point_42['X']) ** 2 + (point_43['Y'] - point_42['Y']) ** 2) +
                               np.sqrt((point_45['X'] - point_44['X']) ** 2 + (point_45['Y'] - point_44['Y']) ** 2) +
                               np.sqrt((point_47['X'] - point_46['X']) ** 2 + (point_47['Y'] - point_46['Y']) ** 2))
                case _:
                    raise ValueError("Invalid opt_recombination value")

            if opt_recombination in [4, 5, 6, 7]:
                point_48 = points[x1]
                point_49 = points[x2]
                point_50 = points[y1]
                point_51 = points[y2]
                point_52 = points[z1]
                point_53 = points[z2]
                old_len = (np.sqrt((point_49['X'] - point_48['X']) ** 2 + (point_49['Y'] - point_48['Y']) ** 2) +
                           np.sqrt((point_51['X'] - point_50['X']) ** 2 + (point_51['Y'] - point_50['Y']) ** 2) +
                           np.sqrt((point_53['X'] - point_52['X']) ** 2 + (point_53['Y'] - point_52['Y']) ** 2))

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
