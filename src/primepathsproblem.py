import logging
import multiprocessing
import pickle
import queue
from typing import Optional

import matplotlib.pyplot as plt

from src.dataprocessing.initialparse import read_cities_data
from src.constants import PRIMES_FILE_PATH, CITY_DATA_FILE_PATH
from src.dataprocessing.chunking import ChunkSpace
import enum

from src.graphtools.bestspanningtree import find_best_spanning_tree
from src.routeoptimization.route import *
from src.constants import *


class PrimePathsProblem:
    def __init__(self):
        self.inter_chunk_connection_trials = 1500
        self.y_chunks = 15
        self.x_chunks = 15
        self.city_count_restriction = 20000
        self.chunk_start_routes = 5
        self.chunk_optimization_iters = 50000
        self.final_opt_routes = 12
        self.final_optimization_iters = 10000000

        # self.inter_chunk_connection_trials = 50
        # self.y_chunks = 5
        # self.x_chunks = 5
        # self.city_count_restriction = 2000
        # self.chunk_start_routes = 3
        # self.chunk_optimization_iters = 100
        # self.final_opt_routes = 4
        # self.final_optimization_iters = 500

        self.chunk_space: Optional[ChunkSpace] = None
        self.cities = None

    def read_problem_data(self, restriction: Optional[int] = None):
        logging.info("Reading problem data")
        self.cities = read_cities_data(CITY_DATA_FILE_PATH, restriction)

    def organize_data_into_chunks(self, x_cells: int, y_cells: int):
        logging.info("Organizing data into chunks")
        self.chunk_space = ChunkSpace(x_cells, y_cells)
        self.chunk_space.put_points(self.cities)
        self.chunk_space.pack_until_at_least_n_per_cell()

    def create_routes_within_chunks(self):
        logging.info("Creating initial routes within chunks")
        chunk_routes = {}

        processing_data = []
        for x_split in range(self.chunk_space.x_cells):
            for y_split in range(self.chunk_space.y_cells):
                chunk = self.chunk_space.chunks[x_split, y_split]
                if chunk is None or len(chunk) == 0:
                    continue
                processing_data.append((chunk, x_split, y_split))

        result = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            result = pool.starmap(self.process_chunk_path, processing_data)

        for chunk_pos, chunk_route_universal in result:
            chunk_routes[chunk_pos] = chunk_route_universal
        return chunk_routes

    def process_chunk_path(self, chunk, x_split, y_split):
        logging.info(f"Processing chunk {x_split, y_split}")
        chunk_route = Route(chunk)
        chunk_route.optimize(Route.three_opt, self.chunk_optimization_iters, self.chunk_start_routes)
        chunk_route_universal = self.translate_route(chunk_route, chunk)
        chunk_pos = (x_split, y_split)
        return chunk_pos, chunk_route_universal

    def create_connections_between_chunks(self, chunk_routes):
        logging.info("Connecting chunks")
        logging.info("Calculating connection point weights")
        connections = {}
        for x_split in range(self.chunk_space.x_cells):
            for y_split in range(self.chunk_space.y_cells):
                this_cell_pos: ChunkPosType = (x_split, y_split)

                if self.chunk_space.chunks[x_split, y_split] is None:
                    logging.info(f"Chunk {this_cell_pos} is empty!")
                    continue
                else:
                    logging.info(f"Processing chunk {this_cell_pos}")
                # neighbour 1
                this_cell = self.chunk_space.chunks[this_cell_pos]

                if x_split < self.chunk_space.x_cells - 1:
                    neighbour_pos: ChunkPosType = (x_split + 1, y_split)
                    neighbour_cell = self.chunk_space.chunks[neighbour_pos]
                    if neighbour_cell is not None:
                        connection_cost, connection_params = self.get_best_connection(this_cell_pos, neighbour_pos,
                                                                                      chunk_routes)
                        connections[(this_cell_pos, neighbour_pos)] = (connection_cost, connection_params)
                    else:
                        logging.info(f"Neighbour of {this_cell_pos}, {neighbour_pos} is empty!")

                # neighbour 2
                if y_split < self.chunk_space.y_cells - 1:
                    neighbour_pos = (x_split, y_split + 1)
                    neighbour_cell = self.chunk_space.chunks[neighbour_pos]
                    if neighbour_cell is not None:
                        connection_cost, connection_params = self.get_best_connection(this_cell_pos, neighbour_pos,
                                                                                      chunk_routes)
                        connections[(this_cell_pos, neighbour_pos)] = (connection_cost, connection_params)
                    else:
                        logging.info(f"Neighbour of {this_cell_pos}, {neighbour_pos} is empty!")

        logging.info("Choosing connection points to form best spanning tree")
        connections_to_keep: list[tuple[ChunkPosType, ChunkPosType]] = find_best_spanning_tree(
            self.make_graph_from_connections(connections),
            lambda edge: connections[edge][0] if edge in connections else 1e18
        )

        logging.info("Applying connections")
        # apply connections
        initial_route = self.merge_routes_in_order(chunk_routes, connections, connections_to_keep)

        # return initial route
        if len(initial_route) < self.city_count_restriction:
            logging.error("Missing cities!")
        return initial_route

    def merge_routes_in_order(self, chunk_routes, connections, connections_to_keep):
        q = queue.Queue()
        good_connection = connections_to_keep.pop()
        q.put(good_connection[0])
        connections_to_keep.add(good_connection)

        initial_route = chunk_routes[good_connection[0]]
        while not q.empty():
            chunk_pos: ChunkPosType = q.get()
            neighbours = self.chunk_space.get_neighbors(chunk_pos)
            for neighbour_pos in neighbours:
                source_merge_city_id, dest_merge_city_id, direction = None, None, None

                if (chunk_pos, neighbour_pos) in connections_to_keep:
                    source_merge_city_id, dest_merge_city_id, direction = connections[(chunk_pos, neighbour_pos)][1]
                    connections_to_keep.remove((chunk_pos, neighbour_pos))
                if (neighbour_pos, chunk_pos) in connections_to_keep:
                    dest_merge_city_id, source_merge_city_id, direction = connections[(neighbour_pos, chunk_pos)][1]
                    connections_to_keep.remove((neighbour_pos, chunk_pos))

                if source_merge_city_id is not None and dest_merge_city_id is not None and direction is not None:
                    initial_route = self.merge_paths(initial_route, chunk_routes[neighbour_pos], source_merge_city_id,
                                                     dest_merge_city_id, direction)
                    q.put(neighbour_pos)
        return initial_route

    def kopt_full_path(self, initial_route: np.array):
        logging.info("Performing Kopt on initial route")

        processing_data = [initial_route for _ in range(self.final_opt_routes)]

        res = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            res = pool.map(self.optimize_instance, processing_data)
        #
        # for data in processing_data:
        #     res.append(self.optimize_instance(data))

        return res

    def optimize_instance(self, initial_route: np.array):
        return Route.two_opt_stat(self.cities, len(self.cities), initial_route, self.final_optimization_iters)

    def solve(self):
        self.read_problem_data(self.city_count_restriction)
        self.organize_data_into_chunks(self.x_chunks, self.y_chunks)
        routes = self.create_routes_within_chunks()
        initial_route = self.create_connections_between_chunks(routes)

        with open("initial_route_cache.bin", "wb") as file:
            pickle.dump(initial_route, file)
        #
        # initial_route = None
        # with open("initial_route_cache.bin", "rb") as file:
        #     initial_route = pickle.load(file)

        results = self.kopt_full_path(initial_route)

        with open("results_cache.bin", "wb") as file:
            pickle.dump(initial_route, file)

        self.plot_results(results)

    def get_best_connection(self, this_pos: ChunkPosType, neighbour_pos: ChunkPosType, chunk_routes: dict):
        this_path = chunk_routes[this_pos]
        neighbour_path = chunk_routes[neighbour_pos]

        this_cost = Route.total_distance(self.cities, this_path)
        neigh_cost = Route.total_distance(self.cities, neighbour_path)

        separated_cost = this_cost + neigh_cost
        best_cost_delta = 1e18
        best_connection = None

        processing_data = []
        for rep in range(self.inter_chunk_connection_trials):
            processing_data.append([this_path, neighbour_path, separated_cost])

        res = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            res = pool.starmap(self.process_random_connection, processing_data)

        best_cost_delta, best_connection = min(res)

        if best_connection == None or best_cost_delta == 0:
            logging.error("Somethin aint right")
        return best_cost_delta, best_connection

    def process_random_connection(self, this_path, neighbour_path, separated_cost):
        try:
            this_it = np.random.randint(low=0, high=len(this_path))
            this_city = this_path[this_it]

            neigh_it = np.random.randint(low=0, high=len(neighbour_path))
            neigh_city = neighbour_path[neigh_it]

            this_shifed = np.append(this_path[-this_it:], this_path[:-this_it])
            neigh_shifed = np.append(neighbour_path[-neigh_it:], neighbour_path[:-neigh_it])

            # 1st possibility, same direction splice
            final_1 = np.append(this_shifed, neigh_shifed)
            final_1_dist = Route.total_distance(self.cities, final_1)
            cost_delta = final_1_dist - separated_cost

            local_best_con = (this_city, neigh_city, 1)
            local_best_cost = cost_delta

            # 2nd possibility, inverded direction
            final_2 = np.append(this_shifed, np.flip(neigh_shifed))
            final_2_dist = Route.total_distance(self.cities, final_2)
            cost_delta = final_2_dist - separated_cost

            if cost_delta < local_best_cost:
                local_best_con = (this_city, neigh_city, -1)
                local_best_cost = cost_delta
        except Exception as e:
            logging.error(f"{e}")
            return 1e18, None

        return local_best_cost, local_best_con

    def merge_paths(self, source_route: np.array, dest_route: np.array, source_city: int, dest_city: int,
                    direction: int):
        source_pos = np.where(source_route == source_city)[0][0]
        dest_pos = np.where(dest_route == dest_city)[0][0]

        source_shifted = np.append(source_route[-source_pos:], source_route[:-source_pos])
        dest_shifted = np.append(dest_route[-dest_pos:], dest_route[:-dest_pos])
        if direction == 1:
            final = np.append(source_shifted, dest_shifted)
        else:
            final = np.append(source_shifted, np.flip(dest_shifted))
        return final

    def translate_route(self, chunk_route, chunk):
        out = []
        for relative_index in chunk_route.best_order:
            out.append(chunk[relative_index]["CityId"])
        return np.array(out, dtype=int)

    def make_graph_from_connections(self, connections) -> Graph:
        g = Graph()
        for connection in connections:
            g.add_edge(connection[0], connection[1])
        return g

    def plot_results(self, results):
        graph_data = []
        for i in range(self.final_optimization_iters):
            graph_data.append(min([result[2][i] for result in results]))

        for result in results:
            plt.plot(result[2])
        plt.show()
