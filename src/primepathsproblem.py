import logging

from src.dataprocessing.initialparse import read_cities_data, read_primes_data
from src.constants import PRIMES_FILE_PATH, CITY_DATA_FILE_PATH
from src.dataprocessing.chunking import ChunkTree
import enum


class CacheEnum(enum.IntEnum):
    NO_CACHE = 0
    CHUNK_TREE_CACHE = 1
    SUB_ROUTES_CACHE = 2
    INITIAL_FULL_ROUTE_CACHE = 3
    FULL_ROUTE_CACHE = 4


class PrimePathsProblem:
    def __init__(self):
        self.chunk_tree = None
        self.cities = None

    def read_problem_data(self):
        self.cities = read_cities_data(CITY_DATA_FILE_PATH)

    def read_cached_data(self) -> CacheEnum:
        pass

    def organize_data_into_chunks(self):
        self.chunk_tree = ChunkTree() # TODO: more of a chunk graph than a tree - binarly bisecting the tree would be really anoying...
        self.chunk_tree.put_points(self.cities)

    def create_routes_within_chunks(self):
        pass

    def create_connections_between_chunks(self):
        pass

    def kopt_full_path(self):
        pass

    def solve(self):
        self.read_problem_data()
        cache = self.read_cached_data()
        # Do only the things that have not been cached yet
        if CacheEnum.CHUNK_TREE_CACHE > cache:
            self.organize_data_into_chunks()
        if CacheEnum.SUB_ROUTES_CACHE > cache:
            self.create_routes_within_chunks()
        if CacheEnum.INITIAL_FULL_ROUTE_CACHE > cache:
            self.create_connections_between_chunks()
        if CacheEnum.FULL_ROUTE_CACHE > cache:
            self.kopt_full_path()
