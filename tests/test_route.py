import math

from src.constants import Point_t
from src.routeoptimization.route import Route
from src.dataprocessing.initialparse import read_cities_data
import numpy as np


# p1 = (0, 0, 0, False)
# p2 = (1, 1, 1, False)
# p3 = (2, 0, 1, True)
# p4 = (3, 1, 0, True)
# p5 = (4, 0, 2, False)
# p6 = (5, 1, 5, True)
# points = [p1, p2, p3, p4, p5, p6]
# np_points = np.array(points, dtype=Point_t)
cities = read_cities_data(filename="../cities.csv", restrict=10, primes_filename="../primes.txt")

def test_route_two_opt_simple():
    route = Route(cities)

    assert route.graph is None
    assert route.n_points == 10
    init_cost = route.best_cost
    order, cost = route.optimize(Route.two_opt, 1, 5)
    assert math.isclose(cost, route.best_cost)
    assert cost < init_cost
    print("Two opt")
    print(init_cost, cost)
    print("_________")


def test_route_three_opt_simple():
    route = Route(cities)

    assert route.graph is None
    assert route.n_points == 10
    init_cost = route.best_cost
    order, cost = route.optimize(Route.three_opt, 1, 5)
    assert math.isclose(cost, route.best_cost)
    assert cost < init_cost
    print("Three opt")
    print(init_cost, cost)
    print("_________")


test_route_two_opt_simple()
test_route_three_opt_simple()
