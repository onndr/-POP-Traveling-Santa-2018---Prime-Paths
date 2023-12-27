import math

from src.constants import Point_t
from src.routeoptimization.route import Route
import numpy as np


def test_route_simple():
    p1 = (0, 0, 0, False)
    p2 = (1, 1, 1, False)
    p3 = (2, 0, 1, True)
    p4 = (3, 1, 0, True)
    points = [p1, p2, p3, p4]
    np_points = np.array(points, dtype=Point_t)

    route = Route(np_points)
    order, cost = route.two_opt(5, 5)

    assert route.n_points == 4
    assert route.best_cost == cost
    assert round(route.best_cost) == 4
    assert route.graph is None
    assert round(route.straight_cost) == 5


test_route_simple()
