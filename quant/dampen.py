'''
Oscillation dampening
https://arxiv.org/abs/2203.11086
'''

import math

from .modules import QConv2d, QLinear


class Dampen:
    """Quadratic pull on the scaled latent weights, u = clamp(w / s, Qn, Qp).

    Writing d = round(u) - u for the signed distance to the nearest grid
    point, the two targets are reflections of each other through
    |d| -> 1/2 - |d|:

        grid      sum_i d_i^2               minimal on the grid
        boundary  sum_i (1/2 - |d_i|)^2     minimal on the decision boundary

    so their forces point opposite ways.  `grid` is the dampening penalty of
    the paper.  `boundary` is the T = diag(m) shadow of the count arm: a
    positive lambda drives weights onto boundaries (flips get cheap), a
    negative one maximizes the flip margin, and ramping lambda from positive
    to negative crossfades exploration into crystallization.

    Summed as in the paper, so lambda keeps its published scale and has to be
    retuned when the weight count changes.  `last` reports the same quantity
    as a mean; both targets lie in [0, 0.25] and equal 1/12 for weights spread
    uniformly over a bin -- read it against that.

    The step size is detached, so the penalty moves the weights rather than
    the grid.  lambda follows a cosine ramp from lam_start to lam over
    training.
    """

    def __init__(self, model, lam, epochs, lam_start=0.0, target='grid'):
        assert target in ('grid', 'boundary')
        self.lam, self.lam_start, self.epochs = lam, lam_start, epochs
        self.target = target

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
        p = sum(self._term(m.wq.u_cache).sum() for m in self.layers)
        self.last = p.detach() / self.numel
        return self.weight(epoch) * p

    def _term(self, u):
        d = u.round() - u
        return d.square() if self.target == 'grid' else (0.5 - d.abs()).square()
