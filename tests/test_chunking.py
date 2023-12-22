import src.dataprocessing.chunking as chunking
from src.constants import Point_t
import numpy as np


def test_simple_chunking():
    points = np.array([
        (1, 0, 0, False),
        (2, 1, 0, True),
        (3, 0, 1, False),
        (4, 1, 1, True),
        (5, 2, 0, 0),
        (6, 1.5, 0.25, 0),
        (7, 1.5, 0.5, 0),
        (8, 1.5, 0.75, 0),
    ], dtype=Point_t)

    chunks = chunking.ChunkSpace(2, 3)
    chunks.put_points(points)
    assert chunks.x_cells == 2
    assert chunks.y_cells == 3
    zero_zero_chunk = chunks.chunks[0, 0]
    assert len(zero_zero_chunk) == 2
    assert 1 in zero_zero_chunk["CityId"]
    assert 2 in zero_zero_chunk["CityId"]
    one_zero_chunk = chunks.chunks[1, 0]
    assert len(one_zero_chunk) == 2
    assert 6 in one_zero_chunk["CityId"]
    assert 5 in one_zero_chunk["CityId"]
    assert chunks.chunks[0, 1] is None
    assert 4 in chunks.chunks[0, 2]["CityId"]
    assert 3 in chunks.chunks[0, 2]["CityId"]


def test_neighbors():
    points = np.array([
        (1, 0, 0, False),
        (2, 1, 0, True),
        (3, 0, 1, False),
        (4, 1, 1, True),
        (5, 2, 0, 0),
        (6, 1.5, 0.25, 0),
        (7, 1.5, 0.5, 0),
        (8, 1.5, 0.75, 0),
    ], dtype=Point_t)

    chunks = chunking.ChunkSpace(2, 3)
    chunks.put_points(points)

    assert chunks.get_neighbors((0, 0)) == {(1, 0), (0, 1)}
    assert chunks.get_neighbors((1, 1)) == {(0, 1), (1, 0), (1, 2)}

    assert len(chunks.get_all_possible_edges()) == 7
    assert chunks.get_all_edges_number_fast() == 7
