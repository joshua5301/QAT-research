'''
Oscillation dampening
https://arxiv.org/abs/2203.11086
'''

import math

from .modules import QConv2d, QLinear


class Dampen:
    """Pull the scaled latent weights toward the nearest grid point.

        sum_i (round(u_i) - u_i)^2,   u = clamp(w / s, Qn, Qp)

    Summed as in the paper, so lambda keeps its published scale and has to be
    retuned when the weight count changes.  `last` reports the same quantity
    as a mean, which lies in [0, 0.25] and equals 1/12 for weights spread
    uniformly over a bin -- read it against that.

    The step size is detached, so the penalty moves the weights rather than
    the grid.  lambda follows a cosine ramp from lam_start to lam over
    training, so the pull is weak while the network is still moving between
    levels and firm once it should settle.
    """

    def __init__(self, model, lam, epochs, lam_start=0.0):
        self.lam, self.lam_start, self.epochs = lam, lam_start, epochs

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'Dampen needs quantized weights (--qat)'
        for m in self.layers:
            m.wq.cache_u = True
        self.numel = sum(m.weight.numel() for m in self.layers)
        self.last = 0.0

    def weight(self, epoch):
        t = min(max(epoch / max(self.epochs - 1, 1), 0.0), 1.0)
        return self.lam_start + (self.lam - self.lam_start) * (1 - math.cos(math.pi * t)) / 2

    def penalty(self, epoch):
        """Call after a forward pass; backward() it after the task loss."""
        assert self.layers[0].wq.u_cache is not None, \
            'penalty() needs a forward pass first'
        p = sum((m.wq.u_cache.round() - m.wq.u_cache).square().sum()
                for m in self.layers)
        self.last = p.detach() / self.numel
        return self.weight(epoch) * p
