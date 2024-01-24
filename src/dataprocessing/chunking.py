from typing import Optional
import numpy as np
from src.constants import Point_t
from src.graphtools.graph import Graph
from src.constants import *

def get_perimiter_points(X, Y, R, x_bounds, y_bounds):
    # Calculate the coordinates of the four corners of the square
    top_left = (X - R // 2, Y + R // 2)
    top_right = (X + R // 2, Y + R // 2)
    bottom_left = (X - R // 2, Y - R // 2)
    bottom_right = (X + R // 2, Y - R // 2)

    # Calculate and print the points on the perimeter
    perimeter_points = []

    # Top side
    for x in range(top_left[0], top_right[0] + 1):
        perimeter_points.append((x, top_left[1]))

    # Right side
    for y in range(top_right[1], bottom_right[1] + 1):
        perimeter_points.append((top_right[0], y))

    # Bottom side
    for x in range(bottom_right[0], bottom_left[0] - 1, -1):
        perimeter_points.append((x, bottom_right[1]))

    # Left side
    for y in range(bottom_left[1], top_left[1] - 1, -1):
        perimeter_points.append((bottom_left[0], y))

    # Remove points outside the specified bounds
    perimeter_points = [(x, y) for x, y in perimeter_points if
                        x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]]

    return perimeter_points


class ChunkSpace:
    MAX_OFFSET = 1.0001

    def __init__(self, x_cells: int, y_cells: int):
        self.chunks: Optional[np.ndarray] = None
        self.x_cells = x_cells
        self.y_cells = y_cells
        self.cell_count = x_cells * y_cells

    def put_points(self, points: np.array):
        self.chunks = np.empty((self.x_cells, self.y_cells), object)
        min_x = np.min(points["X"])
        max_x = np.max(points["X"]) * ChunkSpace.MAX_OFFSET
        min_y = np.min(points["Y"])
        max_y = np.max(points["Y"]) * ChunkSpace.MAX_OFFSET

        for point in points:
            x, y = point["X"], point["Y"]
            x_index = np.floor((x - min_x) / (max_x - min_x) * self.x_cells).astype(int)
            y_index = np.floor((y - min_y) / (max_y - min_y) * self.y_cells).astype(int)

            if self.chunks[x_index, y_index] is not None:
                self.chunks[x_index, y_index] = np.append(self.chunks[x_index, y_index], point)
            else:
                self.chunks[x_index, y_index] = np.array([point], dtype=Point_t)

    def get_neighbors(self, chunk: ChunkPosType):
        out = {(max(0, chunk[0] - 1), chunk[1]),
               (min(self.x_cells - 1, chunk[0] + 1), chunk[1]),
               (chunk[0], max(0, chunk[1] - 1)),
               (chunk[0], min(self.y_cells - 1, chunk[1] + 1))}
        if chunk in out:
            out.remove(chunk)
        return out

    def get_all_possible_edges(self):
        out = []
        for x in range(self.x_cells):
            for y in range(self.y_cells):
                if x < self.x_cells - 1:
                    out.append(((x, y), (x + 1, y)))
                if y < self.y_cells - 1:
                    out.append(((x, y), (x, y + 1)))
        return out

    def get_all_edges_number_fast(self):
        return (self.x_cells - 1) * (self.y_cells - 1) * 2 + (self.x_cells - 1) + (self.y_cells - 1)

    def get_graph(self):
        graph = Graph()
        edges = self.get_all_possible_edges()
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
        return graph

    def pack_until_at_least_n_per_cell(self, n: int = 4):
        for x_split in range(self.x_cells):
            for y_split in range(self.y_cells):
                chunk = self.chunks[x_split, y_split]
                if chunk is None or len(chunk) == 0:
                    continue
                if len(chunk) < n:
                    self.spiral_until_chunk_count_reached(x_split, y_split, n)

    def spiral_until_chunk_count_reached(self, x_split: int, y_split: int, min_count: int = 3):
        gathered_cities = self.chunks[x_split, y_split]
        self.chunks[x_split, y_split] = None
        found = False
        r = 1
        while not found and r < min(self.x_cells, self.y_cells):
            points = get_perimiter_points(x_split, y_split, r, (0, self.x_cells - 1), (0, self.y_cells - 1))

            for point in points:
                perimieter_chunk = self.chunks[point]
                if perimieter_chunk is None or len(perimieter_chunk) == 0:
                    continue
                if len(perimieter_chunk) >= min_count - len(gathered_cities):
                    self.chunks[point] = np.append(self.chunks[point], gathered_cities)
                    found = True
                    break
                else:
                    gathered_cities = np.append(self.chunks[point], gathered_cities)
                    self.chunks[point] = None
            r += 1
