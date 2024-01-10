import math
from src.routeoptimization.route import Route
from src.dataprocessing.initialparse import read_cities_data

n_points = 40
cities = read_cities_data(filename="../cities.csv", restrict=n_points, primes_filename="../primes.txt")


def test_route_two_opt_simple():
    n_iters = 1000
    n_start_routes = 5
    route = Route(cities)

    assert route.graph is None
    assert route.n_points == n_points
    init_cost = route.best_cost
    order, cost = route.optimize(Route.two_opt, n_iters, n_start_routes)
    assert math.isclose(cost, route.best_cost)
    assert cost < init_cost
    print("Two opt")
    print("Initial cost: ", '%.3f' % init_cost)
    print("Post opt cost: ", '%.3f' % cost)
    print("___________________________")


def test_route_three_opt_simple():
    n_iters = 1000
    n_start_routes = 5
    route = Route(cities)

    assert route.graph is None
    assert route.n_points == n_points
    init_cost = route.best_cost
    order, cost = route.optimize(Route.three_opt, n_iters, n_start_routes)
    assert math.isclose(cost, route.best_cost)
    assert cost < init_cost
    print("Three opt")
    print("Initial cost: ", '%.3f' % init_cost)
    print("Post opt cost: ", '%.3f' % cost)
    print("___________________________")


test_route_two_opt_simple()
test_route_three_opt_simple()
