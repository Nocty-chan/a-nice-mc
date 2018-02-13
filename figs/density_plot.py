# Script to visualise density of generated points
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse, os

# Src: https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
def get_density_plot(data, path, iteration):
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
    save_dir= path + '/density_plot/'
    directory = os.path.dirname(save_dir)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    fig.savefig(save_dir + 'density_' + str(iteration) + '.png')
    print ("Saved density plot in %s" % save_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get path to data')
    parser.add_argument('--path', type=str, help='path to a npy file')
    args = parser.parse_args()
    data = np.load(args.path+'/trajectory.npy')
    if (len(data.shape) > 2):
        get_density_plot(data[-1], args.path, 0)
    else:
        get_density_plot(data, args.path, 0)
