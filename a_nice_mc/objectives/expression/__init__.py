import numpy as np
from .. import Energy
from ...utils.evaluation import effective_sample_size, acceptance_rate
from ...utils.logger import save_ess, create_logger

logger = create_logger(__name__)


class Expression(Energy):
    def __init__(self, name='expression', display=True):
        super(Expression, self).__init__()
        self.name = name
        self.display = display
        if display:
            import matplotlib.pyplot as plt
            plt.ion()
        else:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, ncols=1)

    def __call__(self, z):
        raise NotImplementedError(str(type(self)))

    @staticmethod
    def xlim():
        return None

    @staticmethod
    def ylim():
        return None

    def get_transition_matrix(self, data, path):
        def get_class(point):
            norms = np.sum((point-self.means())**2, axis=1)
            return np.argmin(norms)
        if path:
<<<<<<< HEAD
            N, _ = data.shape
            D = self.means().shape[0]
            print D
=======
            N, D = data.shape
>>>>>>> 39a73f39b989deeeb3c770e5e4fc7b14d5adc138
            transition = np.zeros([D, D])
            initial_class = get_class(data[0])
            initial_point = data[0]
            for i in range(1, N):
                new_class = get_class(data[i])
                transition[initial_class][new_class] += 1
                initial_class = new_class
                initial_point = data[i]
            # normalize
            transition = transition / np.sum(transition, axis = -1).reshape([D, 1])
            import matplotlib.pyplot as plt
            import os
            ## Plot transition matrix
            fig_t, ax_t = plt.subplots()
            figplot = ax_t.matshow(transition, interpolation='none', vmin=0, vmax=1)
            plt.colorbar(figplot)
            ind_array = np.arange(0, D, 1)
            x, y = np.meshgrid(ind_array, ind_array)
            for x_val in range(D):
                for y_val in range(D):
                    ax_t.text(x_val, y_val, "{:10.4f}".format(transition[y_val][x_val]), va='center', ha='center')
            save_dir= path + '/density_plot/'
            directory = os.path.dirname(save_dir)
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)
            fig_t.savefig(save_dir + 'transition.png')
            print ("Saved transition in %s" % save_dir)

    def evaluate(self, zv, path=None):
        z, v = zv
        logger.info('Acceptance rate %.4f' % (acceptance_rate(z)))
        z = self.statistics(z)
        ess = effective_sample_size(z, self.mean(), self.std() * self.std(), logger=logger)
        self.get_transition_matrix(z[-1], path)
        if path:
            save_ess(ess, path)
        self.visualize(zv, path)

    def visualize(self, zv, path):
        self.ax1.clear()
        self.ax2.clear()
        z, v = zv
        if path:
            np.save(path + '/trajectory.npy', z)

        z = np.reshape(z, [-1, 2])
        self.ax1.hist2d(z[:, 0], z[:, 1], bins=400)
        self.ax1.set(xlim=self.xlim(), ylim=self.ylim())

        v = np.reshape(v, [-1, 2])
        self.ax2.hist2d(v[:, 0], v[:, 1], bins=400)
        self.ax2.set(xlim=self.xlim(), ylim=self.ylim())

        if self.display:
            import matplotlib.pyplot as plt
            plt.show()
            plt.pause(0.1)
        elif path:
            self.fig.savefig(path + '/visualize.png')
