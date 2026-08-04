import torch
import torch.nn as nn

from .modules import QConv2d, QLinear


class QuantSAM:

    def __init__(self, model, rho=0.05, budget='cost'):
        assert budget in ('cost', 'gain', 'count')
        self.model = model
        self.rho = rho
        self.budget = budget

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'QuantSAM needs quantized weights (--qat)'
        for m in self.layers:
            m.sam = True
            m.wq.cache_round = True

    def _candidates(self, m):
        """ratio, cost, gain, flip delta and a validity mask, all
        weight-shaped."""
        assert m.wq_out is not None and m.wq_out.grad is not None, \
            'ascent_step() needs a backward pass first'
        u, s = m.wq.round_cache
        r = u - u.floor()
        up = r < 0.5

        delta = torch.where(up, s, -s)
        gain = m.wq_out.grad * delta
        cost = ((r - 0.5) * s).square()
        ok = (gain > 0) & (r > 0) & ~(up & (u.floor() >= m.wq.Qp))
        return gain / cost, cost, gain, delta, ok

    def _rank(self, c):
        """The quantity the greedy sorts on.

        Normalizing the ball by T = diag(m), m the distance to the boundary,
        prices every flip at (m/m)^2 = 1, so the count arm ranks on the gain
        alone; the other two rank on gain per unit of squared distance.
        """
        return c[2] if self.budget == 'count' else c[0]

    @torch.no_grad()
    def ascent_step(self):
        cand = [self._candidates(m) for m in self.layers]
        rank = torch.cat([self._rank(c)[c[4]] for c in cand])

        order = torch.argsort(rank, descending=True)
        if self.budget == 'cost':
            cost = torch.cat([c[ok] for _, c, _, _, ok in cand])
            k = int((cost[order].cumsum(0) <= self.rho ** 2).sum())
        elif self.budget == 'gain':
            gain = torch.cat([g[ok] for _, _, g, _, ok in cand])
            k = int((gain[order].cumsum(0) <= self.rho).sum())
        else:
            # uniform price, so the budget is just how many flips fit
            k = min(int(self.rho ** 2), rank.numel())
        cut = rank[order[k - 1]] if k else None

        for m, c in zip(self.layers, cand):
            m.eps = None if cut is None else (c[4] & (self._rank(c) >= cut)) * c[3]
        self._freeze_bn(True)

    @torch.no_grad()
    def restore(self):
        for m in self.layers:
            m.eps = None
            m.wq_out = None
        self._freeze_bn(False)

    def _freeze_bn(self, on):
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if on:
                    m._sam_mom = m.momentum
                m.momentum = 0.0 if on else m._sam_mom
