from a_nice_mc.utils.nice import NiceLayer, NiceNetwork
from a_nice_mc.utils.realnvp import NVPLayer

def create_nice_network(x_dim, v_dim, args):
    net = NiceNetwork(x_dim, v_dim)
    for dims, name, swap in args:
        net.append(NiceLayer(dims, name, swap))
    return net

def create_nvp_network(x_dim, v_dim, args):
    net = NiceNetwork(x_dim, v_dim)
    for dims_s, dims_t, name, swap in args:
        net.append(NVPLayer(dims_s, dims_t, name, swap))
    return net
