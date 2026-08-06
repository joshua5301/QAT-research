'''
SAQ: Sharpness-aware Quantization
https://arxiv.org/abs/2111.12273

SAM with the perturbation applied to the QUANTIZED weights -- the second pass
evaluates Q(w) + e, not Q(w + e), so a perturbation shorter than half a step is
not swallowed by the rounding. The reference does this in
quantize_weight_add_epsilon, i.e. after the rounding, which is what our
QConv2d.eps hook already is.

    e = rho * g / ||g||

with g stacked over dL/dQ(w) and whichever continuous parameters `cont`
selects, a single GLOBAL norm as in the reference. Dropping a group shrinks the
ball to the rest, so the quantized weights get a larger share of rho.

DEFAULT is the reference's own setting: include_bn=True, include_wclip=False,
include_aclip=False, and the biases always in. wscale/ascale are our names for
their two clip values -- perturbing those moves the quantization grid rather
than the weights on it, which is why they are off by default.

Layers left in FP32 by --q-first-last-bits 0 are not in the ball at all: they
have no Q(w) to perturb.
'''
import torch
import torch.nn as nn

from ..lsq import LsqQuantizer
from ..modules import QConv2d, QLinear, freeze_bn


class SAQ:

    CONT = ('bn', 'bias', 'wscale', 'ascale')
    DEFAULT = ('bn', 'bias')

    def __init__(self, model, rho=0.05, cont=DEFAULT):
        cont = set(cont)
        assert cont <= set(self.CONT), sorted(cont - set(self.CONT))
        self.model = model
        self.rho = rho

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'SAQ needs quantized weights (--qat)'
        for m in self.layers:
            m.sam = True

        self.cont = []
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                key = 'bn'
            elif isinstance(m, LsqQuantizer):
                key = 'wscale' if m.is_weight else 'ascale'
            elif isinstance(m, (QConv2d, QLinear)):
                if 'bias' in cont and m.bias is not None:
                    self.cont.append(m.bias)
                continue
            else:
                continue
            if key in cont:
                self.cont += [p for p in m.parameters(recurse=False)
                              if p.requires_grad]
        self._backup = []

    @torch.no_grad()
    def ascent_step(self):
        assert all(m.wq_out is not None and m.wq_out.grad is not None
                   for m in self.layers), 'ascent_step() needs a backward pass first'
        qg = [m.wq_out.grad for m in self.layers]
        cg = [(p, p.grad) for p in self.cont if p.grad is not None]

        norm = torch.norm(torch.stack([g.norm() for g in qg] +
                                      [g.norm() for _, g in cg]))
        scale = self.rho / (norm + 1e-12)

        for m, g in zip(self.layers, qg):
            m.eps = g * scale
        self._backup = [(p, p.data.clone()) for p, _ in cg]
        for p, g in cg:
            p.add_(g * scale)

        freeze_bn(self.model, True)

    @torch.no_grad()
    def restore(self):
        for p, d in self._backup:
            p.data = d
        self._backup = []
        for m in self.layers:
            m.eps = None
            m.wq_out = None
        freeze_bn(self.model, False)
