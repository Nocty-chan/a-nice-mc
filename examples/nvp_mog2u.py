import os
import sys

import numpy as np

sys.path.append(os.getcwd())


def noise_sampler(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])

if __name__ == '__main__':
    from a_nice_mc.objectives.expression.mog2u import MixtureOfGaussians
    from a_nice_mc.models.discriminator import MLPDiscriminator
    from a_nice_mc.models.generator import create_mixed_network
    from a_nice_mc.train.wgan_nll import Trainer

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    mu = np.array([[-5, 0], [5, 0]])
    sigma = np.array([[0.5, 0.5], [1, 1]])
    p = np.array([0.5, 0.5])
    energy_fn = MixtureOfGaussians(mu, sigma, p, display=False, name='mog2-unb')
    discriminator = MLPDiscriminator([400, 400, 400])
    generator = create_mixed_network(
        2, 2,
        [
            ([400], None, 'v1', False),
            ([400], [400], 'x1', True),
            ([400], None, 'v2', False),
        ]
    )

    trainer = Trainer(
        generator,
        energy_fn,
        discriminator,
        noise_sampler,
        b=8,
        m=2,
        mode='real-nvp',
        loss='gan')
    trainer.train()
