'''
QVR: the loss spread a rounding decision costs.

Deployment rounds w onto the grid, so the deployed loss is a random variable
over which side of a boundary each coordinate lands on. That distribution is
not a modelling choice -- it is the deployment process -- so its moments are
directly well posed and nothing has to be borrowed from the PAC-Bayes reading
of SAM. To first order,

    Var_xi[L(Q_xi(w))] = sum_i V(u_i) (g_i s_i)^2 = ||m * g||^2

    m_i = s_i |sin(pi u_i)|        g_i = dL/dQ(w_i)

m is the per-coordinate rounding spread: zero on a grid point, largest on a
boundary, and free of round(). g is what STE already leaves in weight.grad.

So the objective is mean-variance, E[L] + lam * sqrt(Var), and the penalty is
DERIVED rather than relaxed -- no ball, no max, no neighbourhood at all. It is
also the cheap half of it: g is detached, so only the path through m is
differentiated. The curvature term 1/2 tr(Sigma H) of E[L] needs a second
pass and is not taken here.

The step size is detached as well, since s -> 0 would otherwise be a way of
making the penalty vanish without making the model any more robust.
'''
import math

from ..modules import QConv2d, QLinear
from .spread import spread


class QVR:

    FORMS = ('std', 'var')

    def __init__(self, model, lam, epochs, lam_start=0.0, form='std'):
        assert form in self.FORMS, form
        self.lam, self.lam_start, self.epochs = lam, lam_start, epochs
        self.form = form

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'QVR needs quantized weights (--qat)'
        for m in self.layers:
            m.wq.cache_u = True
        self.last = 0.0

    def weight(self, epoch):
        """Cosine from lam_start to lam. Setting lam_start above lam decays
        instead; setting them equal holds lambda constant."""
        t = min(max(epoch / max(self.epochs - 1, 1), 0.0), 1.0)
        return self.lam_start + (self.lam - self.lam_start) * (1 - math.cos(math.pi * t)) / 2

    def penalty(self, epoch):
        """Call after the task backward; backward() it on top."""
        assert self.layers[0].wq.u_cache is not None, \
            'penalty() needs a forward pass first'
        assert self.layers[0].weight.grad is not None, \
            'penalty() needs a backward pass first'
        var = sum(self._term(m) for m in self.layers)
        self.last = float(var.detach().clamp(min=0).sqrt())
        return self.weight(epoch) * (var.clamp(min=1e-24).sqrt()
                                     if self.form == 'std' else var)

    @staticmethod
    def _term(m):
        return (spread(m) * m.weight.grad.detach()).square().sum()
