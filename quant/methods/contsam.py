'''
ContSAM: SAM on the continuous parameters alone, touching no quantized weight.

The control for everything else here. If it matches DiscreteSAM or SAQ, the
quantized half of those balls did nothing and the gain was BN all along -- and
the `share` numbers say that is the default outcome, not the exception, once a
grid-shaped metric shrinks the quantized block.

    cont='bn'        BN affine only
    cont='bn,bias'   exactly SAQ's continuous block, so the tightest control

t matches SAQ's, so the control can be paired with either metric:

    i     T = I,       the plain SAM step, as in SAQ's continuous block
    grid  T = pi |w|,  the Delta -> infinity limit of the rounding spread,
                       i.e. ASAM, as in --saq-t grid

Works without --qat too, which gives plain SAM on the float model.
'''
import torch

from ..modules import freeze_bn
from .cont import CONT, continuous_params
from .spread import spread_cont


class ContSAM:

    CONT = CONT
    DEFAULT = ('bn',)
    T = ('i', 'grid')

    def __init__(self, model, rho=0.05, cont=DEFAULT, t='i'):
        assert t in self.T, t
        self.model = model
        self.rho = rho
        self.t = t

        self.cont = continuous_params(model, cont)
        assert self.cont, 'ContSAM has no parameters to perturb'
        self._backup = []
        self.share = {'quant': 0.0, 'cont': 1.0}

    @torch.no_grad()
    def ascent_step(self):
        cg = [(p, p.grad) for p in self.cont if p.grad is not None]
        assert cg, 'ascent_step() needs a backward pass first'
        tc = [spread_cont(p) if self.t == 'grid' else None for p, _ in cg]

        norm = torch.norm(torch.stack(
            [(g if t is None else t * g).norm() for t, (_, g) in zip(tc, cg)]))
        scale = self.rho / (norm + 1e-12)

        self._backup = [(p, p.data.clone()) for p, _ in cg]
        for (p, g), t in zip(cg, tc):
            p.add_((g if t is None else t.square() * g) * scale)

        freeze_bn(self.model, True)

    @torch.no_grad()
    def restore(self):
        for p, d in self._backup:
            p.data.copy_(d)
        self._backup = []
        freeze_bn(self.model, False)
