import matplotlib.pyplot as plt


def plot_and_save(values, filename, title="Prime Paths problem", x_label="Iterations", y_label="Route cost"):
    plt.xticks(range(len(values)))
    plt.plot(values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()
