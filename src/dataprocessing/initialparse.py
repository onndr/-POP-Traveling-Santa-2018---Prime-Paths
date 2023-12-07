from typing import Optional

from src.constants import (Point_t, PointRaw_t, PRIMES_FILE_PATH, CITY_DATA_FILE_PATH)
import numpy as np

_PRIMES_CACHE: Optional[np.array] = None


def get_primes(filename: str = PRIMES_FILE_PATH) -> np.array:
    if _PRIMES_CACHE:
        return _PRIMES_CACHE
    with open(filename, "r") as file:
        return np.array(list(map(int, file.readline().strip().split(", "))))


def read_cities_data(filename: str = CITY_DATA_FILE_PATH) -> np.array:
    raw_cities = np.genfromtxt(filename, delimiter=',', names=True, dtype=PointRaw_t)
    raw_cities = raw_cities[:10]
    cities = np.zeros(raw_cities.shape, dtype=Point_t)
    for field in PointRaw_t.fields:
        cities[field] = raw_cities[field]

    primes = get_primes()
    j = 0
    for i in range(cities.shape[0]):
        city_id = cities[i]["CityId"]
        while j < len(primes) and city_id >= primes[j]:
            if city_id == primes[j]:
                cities["prime"][i] = True
            j += 1
    return cities
