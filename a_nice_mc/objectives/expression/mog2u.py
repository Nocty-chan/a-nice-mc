import numpy as np
import tensorflow as tf
from a_nice_mc.objectives.expression import Expression
from a_nice_mc.utils.logger import create_logger

logger = create_logger(__name__)


## Mixture of MixtureOfGaussians
## mu_1 = (-5, 0) sigma_1 = (0.5, 0.5)
## mu_2 = (5, 0) sigma_2 = (1, 1)
mu_1 = np.array([-5, 0])
mu_2 = np.array([5, 0])
sigma_1 = np.array([0.5, 0.5])
sigma_2 = np.array([1, 1])
p = 0.5
class MixtureOfGaussians(Expression):
    def __init__(self, name='mog2-unb', display=True):
        super(MixtureOfGaussians, self).__init__(name=name, display=display)
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')

    def __call__(self, z):
        z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
        z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
        v1 = (z1 - mu_1[0])**2 / (2 * sigma_1[0]**2) \
            + (z2 - mu_1[1])**2 / (2 * sigma_1[1]**2)
        v2 = (z1 - mu_2[0])**2 / (2 * sigma_2[0]**2) \
            + (z2 - mu_2[1])**2 / (2 * sigma_2[1]**2)
        pdf1 = tf.exp(-v1) / (2 * np.pi * sigma_1[0] * sigma_1[1])
        pdf2 = tf.exp(-v2) / (2 * np.pi * sigma_2[0] * sigma_2[1])
        return -tf.log(0.5 * pdf1 + 0.5 * pdf2)

    @staticmethod
    def means():
        return np.stack([mu_1, m_2], axis=0)
    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    @staticmethod
    def stds():
        return np.stack([sigma_1, sigma_2], axis=0)

    @staticmethod
    def std():
        # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        std = p * sigma_1**2 + (1.0 - p) * sigma_2**2 + p * (1.0 - p) * (mu_1 - mu_2)**2
        return np.sqrt(std)

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-10, 10]

    @staticmethod
    def ylim():
        return [-8, 8]
