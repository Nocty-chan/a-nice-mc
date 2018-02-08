import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import argparse

rc('animation', html='html5')
def init():
    sca.set_data([], [])
    return (sca,)

def animate(i):
    x = data[:i, 0]
    y = data[:i, 1]
    sca.set_data(x, y)
    return (sca,)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get path to data')
    parser.add_argument('--path', type=str, help='path to a npy file')
    parser.add_argument('--frames', type=int, default=1000, help='number of frames')
    parser.add_argument('--interval', type=int, default=10, help='interval in ms between two frames.')
    args = parser.parse_args()

    data = np.load(args.path)
    if (len(data.shape) > 2):
        data = data[-1]

    fig, ax = plt.subplots()
    ax.set_xlim((-10, 10))
    ax.set_ylim((-8, 8))
    sca, = ax.plot([], [], 'x')
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=args.frames, interval=args.interval, blit=True)
    plt.show()
