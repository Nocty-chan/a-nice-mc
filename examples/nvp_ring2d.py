import os
import sys

import numpy as np

sys.path.append(os.getcwd())


def noise_sampler(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])

if __name__ == '__main__':
    from a_nice_mc.objectives.expression.ring2d import Ring2d
    from a_nice_mc.models.discriminator import MLPDiscriminator
    from a_nice_mc.models.generator import create_mixed_network
    from a_nice_mc.train.wgan_nll import Trainer

    energy_fn = Ring2d(display=False)
    discriminator = MLPDiscriminator([400, 400, 400])
    generator = create_mixed_network(
        2, 2,
        [
            ([400], None, 'v1', False),
            ([400],[400], 'x1', True),
            ([400], None, 'v2', False),
        ]
    )

    trainer = Trainer(generator, energy_fn, discriminator, noise_sampler, b=8, m=2, mode='real-nvp')
    trainer.train()
