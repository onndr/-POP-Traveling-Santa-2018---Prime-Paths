from typing import Tuple

import numpy as np

PointRaw_t = np.dtype([('CityId', int), ('X', float), ('Y', float)])
Point_t = np.dtype([('CityId', int), ('X', float), ('Y', float), ('prime', bool)])

ChunkPosType = Tuple[int, int]

CITY_DATA_FILE_PATH = 'cities.csv'
PRIMES_FILE_PATH = 'primes.txt'
