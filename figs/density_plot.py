# Script to visualise density of generated points
# Src: https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse

def get_plot(data):
    x = data[:, 0]
    y = data[:, 1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    fig, ax = plt.subplots()
    ax.set_xlim((-10, 10))
    ax.set_ylim((-8, 8))
    axsc = ax.scatter(x, y, c=z, s=5, edgecolor='')
    plt.colorbar(axsc)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get path to data')
    parser.add_argument('--path', type=str, help='path to a npy file')
    args = parser.parse_args()

    data = np.load(args.path)
    if (len(data.shape) > 2):
        data = data[-1]

    get_plot(data)
