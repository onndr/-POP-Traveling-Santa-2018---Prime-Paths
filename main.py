import pandas as pd
from matplotlib import pyplot as plt

DATA_FILE = "./cities.csv"


def main():
    cities_df = pd.read_csv(DATA_FILE)
    X_max_value = cities_df['X'].max()
    X_min_value = cities_df['X'].min()
    Y_max_value = cities_df['Y'].max()
    Y_min_value = cities_df['Y'].min()
    plt.scatter(cities_df['X'], cities_df['Y'], marker=".")
    plt.show()


if __name__ == "__main__":
    main()
