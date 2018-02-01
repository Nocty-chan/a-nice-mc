import numpy as np
import tensorflow as tf
from a_nice_mc.objectives.expression import Expression
from a_nice_mc.utils.logger import create_logger

logger = create_logger(__name__)


## Mixture of MixtureOfGaussians
## mu_1 = (-5, 0) sigma_1 = (1, 1)
## mu_2 = (5, 0) sigma_2 = (2, 2)
class MixtureOfGaussians(Expression):
    def __init__(self, name='mog2-unb', display=True):
        super(MixtureOfGaussians, self).__init__(name=name, display=display)
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')

    def __call__(self, z):
        z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
        z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
        v1 = tf.sqrt((z1 - 5) * (z1 - 5) + z2 * z2)
        v2 = tf.sqrt((z1 + 5) * (z1 + 5) + z2 * z2) * 0.5
        pdf1 = tf.exp(-0.5 * v1 * v1) / tf.sqrt(2 * np.pi)
        pdf2 = tf.exp(-0.5 * v2 * v2) / tf.sqrt(2 * np.pi * 4)
        return -tf.log(0.5 * pdf1 + 0.5 * pdf2)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    @staticmethod
    def std():
        # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        return np.array([np.sqrt(27.5), np.sqrt(2.5)])

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-10, 10]

    @staticmethod
    def ylim():
        return [-8, 8]
