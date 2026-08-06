'''
AOQ: Allowing Oscillation Quantization
Xie et al., ICCV 2025 -- https://github.com/muzenc/AOQ

Three stages. The paper puts them at 1/5, 2/5 and 2/5 of training, i.e.
boundaries at 0.2 and 0.6, which is also where the released code hardcodes
them (epochs 50 and 150 of 250). Kept as fractions so they follow --epochs:

    explore  [0, s1)    alpha contracts the grid from 1 to alpha_min along a
                        cosine, so weights cross boundaries and the run visits
                        more quantized configurations
    settle   [s1, s2)   alpha released to 1; the learned step size takes over
    dampen   [s2, 1]    the oscillation penalty pulls weights back onto the
                        grid so the run can converge

ADAPTATION: the paper contracts TWO scales in the first stage, s_th for the
thresholds and s_le for the levels, both to 30-35% of their initial value --
that decoupling is its point. Uniform LSQ has a single s playing both roles, so
alpha contracts it once. Same weights cross the same kind of boundary, but it
is not the same quantizer; a faithful port needs a second quantizer class.

The reference weights its penalty by 0.01 against a KD loss; that constant does
not transfer, since our penalty is the Qualcomm one in weight units. Treat
damp_lam as its own knob.

Its recipe does not transfer either: batch 512 (256 in the released scripts),
weight decay 0, and knowledge distillation throughout. Hold your own recipe
fixed across arms instead, or the comparison is confounded.
'''
import math

from ..modules import QConv2d, QLinear
from .ooq import OOQ


class AOQ:

    def __init__(self, model, epochs, stage1=0.2, stage2=0.6,
                 alpha_min=0.3, damp_lam=0.01):
        assert 0.0 <= stage1 <= stage2 <= 1.0, (stage1, stage2)
        assert 0.0 < alpha_min <= 1.0, alpha_min
        self.epochs, self.stage1, self.stage2 = epochs, stage1, stage2
        self.alpha_min = alpha_min

        self.wq = [m.wq for m in model.modules()
                   if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.wq, 'AOQ needs quantized weights (--qat)'

        # lambda is held flat: the stage boundary already gates it
        self.damp = OOQ(model, damp_lam, epochs,
                        lam_start=damp_lam, anneal_start=0.0)
        self.last = 0.0

    def stage(self, epoch):
        t = epoch / max(self.epochs, 1)
        return 1 if t < self.stage1 else 2 if t < self.stage2 else 3

    def alpha(self, epoch):
        if self.stage(epoch) != 1:
            return 1.0
        t = epoch / max(self.stage1 * self.epochs, 1)
        return (1 + self.alpha_min) / 2 + (1 - self.alpha_min) / 2 * math.cos(math.pi * t)

    def set_epoch(self, epoch):
        """Call once per epoch, before training."""
        a = self.alpha(epoch)
        for q in self.wq:
            q.alpha = a

    def weight(self, epoch):
        return self.damp.lam if self.stage(epoch) == 3 else 0.0

    def penalty(self, epoch):
        """None outside the dampening stage."""
        if self.stage(epoch) != 3:
            self.last = 0.0
            return None
        p = self.damp.penalty(epoch)
        self.last = self.damp.last
        return p
