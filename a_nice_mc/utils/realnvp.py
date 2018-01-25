import tensorflow as tf

from a_nice_mc.utils.nice import Layer
from a_nice_mc.utils.layers import dense

class NVPLayer(Layer):
    def __init__(self, dims_s, dims_t, name='nvp', swap=False):
        """
        Real NVPs Layer that takes in [x, v] as input and updates one of them.
        The Jacobian for the NVP Layer only depends on the scaling factor
        :param dims_s/dims_t: structure of the two NVP networks
        :param name: TensorFlow variable name scope for variable reuse.
        :param swap: Update x if True, or update v if False.
        """
        super(NVPLayer, self).__init__()
        self.dims_s, self.dims_t, self.reuse, self.swap = dims_s, dims_t, False, swap
        self.name = 'generator/' + name

    def forward(self, inputs):
        x, v = inputs
        x_dim, v_dim = x.get_shape().as_list()[-1], v.get_shape().as_list()[-1]
        if self.swap:
            t = self.add(v, x_dim, 'translate', reuse=self.reuse)
            s = self.add(v, x_dim, 'scale', reuse=self.reuse)
            x = tf.multiply(x, tf.exp(s)) + t
            return [x, v], tf.exp(tf.reduce_sum(s, 1))
        else:
            t = self.add(x, v_dim, 'translate', reuse=self.reuse)
            s = self.add(x, v_dim, 'scale', reuse=self.reuse)
            v = tf.multiply(v, tf.exp(s)) + t
            return [x, v], tf.reduce_sum(s, 1)

    def backward(self, inputs):
        x, v, = inputs
        x_dim, v_dim = x.get_shape().as_list()[-1], v.get_shape().as_list()[-1]
        if self.swap:
            t = self.add(v, x_dim, 'translate', reuse=True)
            s = self.add(v, x_dim, 'scale', reuse=True)
            x = tf.multiply(x - t, tf.exp(-s))
            return [x, v], tf.exp(-tf.reduce_sum(s, 1))
        else:
            t = self.add(x, v_dim, 'translate', reuse=True)
            s = self.add(x, v_dim, 'scale', reuse=True)
            v = tf.multiply(v - t, tf.exp(-s))
            return [x, v], -tf.reduce_sum(s, 1)

    def add(self, x, dx, name, reuse=False):
        dims = None
        if name == 'translate':
            dims = self.dims_t
        if name == 'scale':
            dims = self.dims_s

        with tf.variable_scope(self.name+name, reuse=reuse):
            for dim in dims:
                x = dense(x, dim, activation_fn=tf.nn.relu)
            x = dense(x, dx)
            return x

    def create_variables(self, x_dim, v_dim):
        assert not self.reuse
        x = tf.zeros([1, x_dim])
        v = tf.zeros([1, v_dim])
        _ = self.forward([x, v])
        self.reuse = True
