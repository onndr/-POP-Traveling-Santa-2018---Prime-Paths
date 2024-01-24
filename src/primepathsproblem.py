import logging
import queue
from typing import Optional

from src.dataprocessing.initialparse import read_cities_data
from src.constants import PRIMES_FILE_PATH, CITY_DATA_FILE_PATH
from src.dataprocessing.chunking import ChunkSpace
import enum

from src.graphtools.bestspanningtree import find_best_spanning_tree
from src.routeoptimization.route import *
from src.constants import *


class PrimePathsProblem:
    def __init__(self):
        self.y_chunks = 3
        self.x_chunks = 3
        self.city_count_restriction = 250
        self.chunk_start_routes = 5
        self.chunk_optimization_iters = 20
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
        for x_split in range(self.chunk_space.x_cells):
            for y_split in range(self.chunk_space.y_cells):
                chunk = self.chunk_space.chunks[x_split, y_split]
                if chunk is None or len(chunk) == 0:
                    continue
                chunk_route = Route(chunk)
                chunk_route.optimize(
                    Route.three_opt, self.chunk_optimization_iters, self.chunk_start_routes
                )
                chunk_route_universal = self.translate_route(chunk_route, chunk)
                chunk_routes[(x_split, y_split)] = chunk_route_universal
        return chunk_routes

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
                # neighbour 1
                this_cell = self.chunk_space.chunks[this_cell_pos]

                if x_split < self.chunk_space.x_cells - 1:
                    neighbour_pos: ChunkPosType = (x_split + 1, y_split)
                    neighbour_cell = self.chunk_space.chunks[neighbour_pos]
                    if neighbour_cell is not None:
                        logging.info(f"Neighbour of {this_cell_pos}, {neighbour_pos} is empty!")
                        connection_cost, connection_params = self.get_best_connection(this_cell_pos, neighbour_pos,
                                                                                      chunk_routes)
                        connections[(this_cell_pos, neighbour_pos)] = (connection_cost, connection_params)

                # neighbour 2
                if y_split < self.chunk_space.y_cells - 1:
                    neighbour_pos = (x_split, y_split + 1)
                    neighbour_cell = self.chunk_space.chunks[neighbour_pos]
                    if neighbour_cell is not None:
                        connection_cost, connection_params = self.get_best_connection(this_cell_pos, neighbour_pos,
                                                                                      chunk_routes)
                        connections[(this_cell_pos, neighbour_pos)] = (connection_cost, connection_params)

        logging.info("Choosing connection points to form best spanning tree")
        connections_to_keep: list[tuple[ChunkPosType, ChunkPosType]] = find_best_spanning_tree(
            self.chunk_space.get_graph(), lambda edge: connections[edge][0] if edge in connections else 1e18
        )

        logging.info("Applying connections")
        # apply connections
        initial_route = self.merge_routes_in_order(chunk_routes, connections, connections_to_keep)

        # return initial route
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

    def kopt_full_path(self, initial_route: list):
        logging.info("Performing Kopt on initial route")
        pass

    def solve(self):
        self.read_problem_data(self.city_count_restriction)
        # Do only the things that have not been cached yet
        self.organize_data_into_chunks(self.x_chunks, self.y_chunks)
        routes = self.create_routes_within_chunks()
        initial_route = self.create_connections_between_chunks(routes)
        self.kopt_full_path(initial_route)

    def get_best_connection(self, this_pos: ChunkPosType, neighbour_pos: ChunkPosType, chunk_routes: dict):
        this_path = chunk_routes[this_pos]
        neighbour_path = chunk_routes[neighbour_pos]

        this_cost = Route.total_distance(self.cities, this_path)
        neigh_cost = Route.total_distance(self.cities, neighbour_path)

        separated_cost = this_cost + neigh_cost
        best_cost_delta = 0
        best_connection = None

        for this_it, this_city in enumerate(this_path):
            for neigh_it, neigh_city in enumerate(neighbour_path):
                this_shifed = np.append(this_path[-this_it:], this_path[:-this_it])
                neigh_shifed = np.append(neighbour_path[-neigh_it:], neighbour_path[:-neigh_it])

                # 1st possibility, same direction splice
                final_1 = np.append(this_shifed, neigh_shifed)
                final_1_dist = Route.total_distance(self.cities, final_1)
                cost_delta = separated_cost - final_1_dist
                if cost_delta > best_cost_delta:
                    best_connection = (this_city, neigh_city, 1)
                    best_cost_delta = cost_delta

                # 2nd possibility, inverded direction
                final_2 = np.append(this_shifed, np.flip(neigh_shifed))
                final_2_dist = Route.total_distance(self.cities, final_2)
                cost_delta = separated_cost - final_2_dist
                if cost_delta > best_cost_delta:
                    best_connection = (this_city, neigh_city, -1)
                    best_cost_delta = cost_delta

        return best_cost_delta, best_connection

    def merge_paths(self, source_route: np.array, dest_route: np.array, source_city: int, dest_city: int, direction: int):
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