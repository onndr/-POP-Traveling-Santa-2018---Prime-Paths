import logging

from src.primepathsproblem import PrimePathsProblem


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    problem = PrimePathsProblem()
    problem.solve()


if __name__ == "__main__":
    main()
