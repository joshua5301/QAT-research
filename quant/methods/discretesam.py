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

rho -> 0 reduces to nearest rounding. The flips live in QConv2d.eps rather
than in the weights, so restoring them is only a matter of clearing the hooks.

Whichever continuous parameters `cont` selects draw on the SAME budget. Buying
G nats out of that block costs ||G g_c / ||g_c||^2||^2 = G^2 / ||g_c||^2, so
its marginal price rises with G while a flip at ratio r has the flat price
1 / r. Equalizing the two -- one Lagrange multiplier over both blocks -- gives

    G_c = ||g_c||^2 / (2 r*)        e_c = g_c / (2 r*)

with r* the ratio at which the greedy stops. The block therefore has no knob
of its own: it simply keeps buying until it is as expensive as the next flip.
Deselecting it leaves ||g_c|| = 0 and the discrete-only rule is recovered.
'''
import torch

from ..modules import QConv2d, QLinear, freeze_bn
from .cont import CONT, DEFAULT, continuous_params


class DiscreteSAM:

    CONT = CONT
    DEFAULT = DEFAULT

    def __init__(self, model, rho=0.05, cont=DEFAULT):
        self.model = model
        self.rho = rho

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'DiscreteSAM needs quantized weights (--qat)'
        for m in self.layers:
            m.sam = True
            m.wq.cache_round = True

        self.cont = continuous_params(model, cont)
        self._backup = []

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

        cg = [(p, p.grad) for p in self.cont if p.grad is not None]
        gc2 = float(sum(g.pow(2).sum() for _, g in cg)) if cg else 0.0

        order = torch.argsort(rank, descending=True)
        # the continuous share is what that block buys at the marginal flip's
        # price, so both blocks are spent against the one budget
        spent = gain[order].cumsum(0) + gc2 / (2 * rank[order])
        k = int((spent <= self.rho).sum())
        cut = rank[order[k - 1]] if k else None

        for m, (rt, _, delta, ok) in zip(self.layers, cand):
            m.eps = None if cut is None else (ok & (rt >= cut)) * delta

        # no flip is affordable, so the block takes the budget on its own
        scale = (1.0 / (2 * float(cut)) if cut is not None else
                 self.rho / gc2 if gc2 > 0 else 0.0)
        self._backup = [(p, p.data.clone()) for p, _ in cg]
        for p, g in cg:
            p.add_(g * scale)
        freeze_bn(self.model, True)

    @torch.no_grad()
    def restore(self):
        for p, d in self._backup:
            p.data.copy_(d)
        self._backup = []
        for m in self.layers:
            m.eps = None
            m.wq_out = None
        freeze_bn(self.model, False)
