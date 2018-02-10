import numpy as np
import tensorflow as tf
from a_nice_mc.objectives.expression import Expression
from a_nice_mc.utils.logger import create_logger

logger = create_logger(__name__)


class MixtureOfGaussians(Expression):
    ''' Defines a MixtureOfGaussians distribution
    :param mu: means of each gaussian,  N x D where N is the number of gaussians
    :param sigma: variances of each gaussian, N x D
    :param p: probabilities of each gaussian, N x 1, must sum to 1.
    '''
    def __init__(self, mu, sigma, p, name='mog2', display=True):
        super(MixtureOfGaussians, self).__init__(name=name, display=display)
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')
        self.mu = mu
        self.sigma = sigma
        self.modes = self.mu.shape[0]
        self.dim = self.mu.shape[1]
        self.p = p
        self.mean_value = None
        self.std_value = None
        print self.modes

    def __call__(self, z):
        z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
        z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
        v = [(z1 - self.mu[i][0])**2 / (2 * self.sigma[i][0]**2) \
            + (z2 - self.mu[i][1])**2 / (2 * self.sigma[i][1]**2) for i in range(self.modes)]
        pdf = [self.p[i] * (tf.exp(-v[i]) / (2 * np.pi * np.prod(self.sigma[i]))) for i in range(self.modes)]
        return -tf.log(tf.reduce_sum(pdf, axis=0))

    def means(self):
        return self.mu

    def mean(self):
        if self.mean_value is None:
            self.mean_value = np.sum(self.p * self.mu, axis=0)
        return self.mean_value

    def stds(self):
        return self.sigma

    def std(self):
        # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        if self.std_value is None:
            term1 = np.sum(self.p * self.sigma * self.sigma, axis=0)
            term2 = np.sum(self.p * self.mu * self.mu, axis=0)
            self.std_value = term1 + term2 - self.mean() * self.mean()
        return np.sqrt(self.std_value)

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-10, 10]

    @staticmethod
    def ylim():
        return [-8, 8]
