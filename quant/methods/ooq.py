'''
OOQ: Overcoming Oscillations in Quantization-aware training
https://arxiv.org/abs/2203.11086
'''

import math

from ..modules import QConv2d, QLinear


class OOQ:
    """Quadratic pull of the latent weights onto the grid.

    Matched to the reference implementation, since lambda only carries its
    published scale if all three of these do:

      units        (Q(w) - w)^2, in WEIGHT units, not in units of the step;
                   measuring (round(u) - u)^2 instead inflates it by 1/s^2,
                   ~800x at 4 bit and varying per channel
      aggregation  sum over output channels, then mean -- their 'kernel_mean'
                   default. Note the paper writes ||.||_F^2, a plain sum, which
                   is larger again by the per-channel element count; that is
                   also why its ablation tops out at 1e-2 while the released
                   script anneals to 0.1
      schedule     lambda is held at lam_start for the first anneal_start of
                   training, then cosine-ramped to lam. The hold is an
                   implementation detail, not something the paper states

    Clipped weights contribute nothing, since Q(w) meets w at the clip bound.
    The step size is detached, so the penalty moves the weights, not the grid.
    """

    def __init__(self, model, lam, epochs, lam_start=0.0, anneal_start=0.25):
        self.lam, self.lam_start, self.epochs = lam, lam_start, epochs
        self.anneal_start = anneal_start

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'OOQ needs quantized weights (--qat)'
        for m in self.layers:
            m.wq.cache_u = True
        self.last = 0.0

    def weight(self, epoch):
        t0 = self.anneal_start * self.epochs
        if epoch < t0:
            return self.lam_start
        t = min((epoch - t0) / max(self.epochs - t0, 1), 1.0)
        return self.lam_start + (self.lam - self.lam_start) * (1 - math.cos(math.pi * t)) / 2

    def penalty(self, epoch):
        """Call after a forward pass; backward() it after the task loss."""
        assert self.layers[0].wq.u_cache is not None, \
            'penalty() needs a forward pass first'
        p = sum(self._term(*m.wq.u_cache) for m in self.layers)
        self.last = float(p.detach())
        return self.weight(epoch) * p

    @staticmethod
    def _term(u, s):
        return (((u.round() - u) * s).square()).sum(0).mean()
