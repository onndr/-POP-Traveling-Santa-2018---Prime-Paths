from typing import Optional
import numpy as np
from src.constants import Point_t


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

            if self.chunks[x_index, y_index]:
                self.chunks[x_index, y_index] = np.append(self.chunks[x_index, y_index], point)
            else:
                self.chunks[x_index, y_index] = np.array([point], dtype=Point_t)

    def get_neighbors(self, chunk: tuple[int, int]):
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
