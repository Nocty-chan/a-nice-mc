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
        data = data[0]

    get_plot(data)


# # Libraries
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import kde
#
# # Create data: 200 points
# data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
# x, y = data.T
#
# # Create a figure with 6 plot areas
# fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))
#
# # Everything sarts with a Scatterplot
# axes[0].set_title('Scatterplot')
# axes[0].plot(x, y, 'ko')
# # As you can see there is a lot of overplottin here!
#
# # Thus we can cut the plotting window in several hexbins
# nbins = 20
# axes[1].set_title('Hexbin')
# axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)
#
# # 2D Histogram
# axes[2].set_title('2D Histogram')
# axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)
#
# # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
# k = kde.gaussian_kde(data.T)
# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#
# # plot a density
# axes[3].set_title('Calculate Gaussian KDE')
# axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)
#
# # add shading
# axes[4].set_title('2D Density with shading')
# axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
#
# # contour
# axes[5].set_title('Contour')
# axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
# axes[5].contour(xi, yi, zi.reshape(xi.shape) )
#
# plt.show()
