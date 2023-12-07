from typing import Optional

import numpy as np


class ChunkNode:
    def __init__(self, parent):
        self.parent: Optional[ChunkNode] = parent
        self.points: Optional[np.array] = None
        self.children: list[list[ChunkNode]] = [[], []]
        self.split_x: Optional[float] = None
        self.split_y: Optional[float] = None


class ChunkTree:
    def __init__(self):
        self.root = ChunkNode(None)

    def put_points(self, points: np.array):
        self._point_splitter_recursive(points, self.root)

    def _point_splitter_recursive(self, points: np.array, current_node: ChunkNode):
        pass
