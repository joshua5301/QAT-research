import torch
import torch.nn as nn

from .modules import QConv2d, QLinear


class QuantSAM:

    def __init__(self, model, rho=0.05):
        self.model = model
        self.rho = rho

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'QuantSAM needs quantized weights (--qat)'
        for m in self.layers:
            m.sam = True
            m.wq.cache_round = True

    def _candidates(self, m):
        """ratio, cost, flip delta and a validity mask, all weight-shaped."""
        assert m.wq_out is not None and m.wq_out.grad is not None, \
            'ascent_step() needs a backward pass first'
        u, s = m.wq.round_cache
        r = u - u.floor()
        up = r < 0.5

        delta = torch.where(up, s, -s)
        gain = m.wq_out.grad * delta
        cost = ((r - 0.5) * s).square()
        ok = (gain > 0) & (r > 0) & ~(up & (u.floor() >= m.wq.Qp))
        return gain / cost, cost, delta, ok

    @torch.no_grad()
    def ascent_step(self):
        cand = [self._candidates(m) for m in self.layers]
        ratio = torch.cat([rt[ok] for rt, _, _, ok in cand])
        cost = torch.cat([c[ok] for _, c, _, ok in cand])

        order = torch.argsort(ratio, descending=True)
        k = int((cost[order].cumsum(0) <= self.rho ** 2).sum())
        cut = ratio[order[k - 1]] if k else None

        for m, (rt, _, delta, ok) in zip(self.layers, cand):
            m.eps = None if cut is None else (ok & (rt >= cut)) * delta
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
