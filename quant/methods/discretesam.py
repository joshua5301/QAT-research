'''
DiscreteSAM: SAM whose neighbourhood is the quantization grid itself.

A continuous ball is the natural neighbourhood while weights are continuous.
Once they are quantized the reachable models form a discrete set, yet prior
work keeps perturbing continuously -- SAQ evaluates Q(w) + e, which is not a
model anyone can deploy. Here the ascent may only change rounding decisions, so
every point in the neighbourhood is a realizable quantized model.

With u = w/s and r = frac(u), per quantized coordinate:

    gain = g * d                  d = +-s, the flip to the other neighbour
    cost = (|r - 1/2| * s)^2      squared latent distance to the boundary

    max_S  sum_S gain    s.t.  sum_S gain <= rho

The budget is on the first-order loss increase itself, so rho is read in nats
and does not depend on the step size or the bit width. Any selection reaching
the budget attains the same total gain, so the ordering only decides WHICH
flips get there: sorting by gain / cost buys the target loss increase with the
least latent movement.

rho -> 0 reduces to nearest rounding. Continuous parameters (BN, biases, step
sizes) are never perturbed, and the flips live in QConv2d.eps rather than in
the weights, so restoring is only a matter of clearing the hooks.
'''
import torch

from ..modules import QConv2d, QLinear, freeze_bn


class DiscreteSAM:

    def __init__(self, model, rho=0.05):
        self.model = model
        self.rho = rho

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'DiscreteSAM needs quantized weights (--qat)'
        for m in self.layers:
            m.sam = True
            m.wq.cache_round = True

    def _candidates(self, m):
        """ratio, gain, flip delta and a validity mask, all weight-shaped."""
        assert m.wq_out is not None and m.wq_out.grad is not None, \
            'ascent_step() needs a backward pass first'
        u, s = m.wq.round_cache
        r = u - u.floor()
        up = r < 0.5                       # nearest is the floor, so flip up

        delta = torch.where(up, s, -s)
        gain = m.wq_out.grad * delta
        cost = ((r - 0.5) * s).square()
        # r == 0 is clamped or exactly on a grid point; flipping up out of the
        # top level is not representable
        ok = (gain > 0) & (r > 0) & ~(up & (u.floor() >= m.wq.Qp))
        return gain / cost, gain, delta, ok

    @torch.no_grad()
    def ascent_step(self):
        cand = [self._candidates(m) for m in self.layers]
        rank = torch.cat([rt[ok] for rt, _, _, ok in cand])
        gain = torch.cat([g[ok] for _, g, _, ok in cand])

        order = torch.argsort(rank, descending=True)
        k = int((gain[order].cumsum(0) <= self.rho).sum())
        cut = rank[order[k - 1]] if k else None

        for m, (rt, _, delta, ok) in zip(self.layers, cand):
            m.eps = None if cut is None else (ok & (rt >= cut)) * delta
        freeze_bn(self.model, True)

    @torch.no_grad()
    def restore(self):
        for m in self.layers:
            m.eps = None
            m.wq_out = None
        freeze_bn(self.model, False)
