'''
LSQ: Learned Step Size Quantization
https://arxiv.org/abs/1902.08153
'''

import math

import torch
import torch.nn as nn


def grad_scale(x, s):
    """
    Forward : x
    Backward: gradient multiplied by s
    """
    return x.detach() + (x - x.detach()) * s

def round_pass(x):
    """
    Forward: round(x)
    Backward: identity (straight-through)
    """
    return x.round().detach() + (x - x.detach())

class LsqQuantizer(nn.Module):
    """
    Fake-quantizer with a learnable step size.
    """

    def __init__(self, bits, is_weight, per_channel=False, num_channels=None,
                 signed_mode='auto', init_mode='lsq'):
        super(LsqQuantizer, self).__init__()
        assert bits >= 2, 'bits must be >= 2'
        assert not (per_channel and num_channels is None)
        assert signed_mode in ('auto', 'always')
        assert init_mode in ('lsq', 'minmax')

        self.bits = bits
        self.is_weight = is_weight
        self.per_channel = per_channel
        self.signed_mode = signed_mode
        self.init_mode = init_mode

        self.s = nn.Parameter(torch.ones(num_channels if per_channel else 1))

        self.signed = 1 if (is_weight or signed_mode == 'always') else 0
        self._ready = False

        self.cache_round = False    # GridSAM needs (u, step) from the forward
        self.round_cache = None
        self.cache_u = False        # dampening needs u with the grid held fixed
        self.u_cache = None

        self._set_levels()

    def _set_levels(self):
        if self.signed:
            self.Qn = -2 ** (self.bits - 1)
            self.Qp = 2 ** (self.bits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** self.bits - 1

    def _step_from_stats(self, mean_abs, max_abs):
        if self.init_mode == 'minmax':
            return max_abs / self.Qp
        return 2.0 * mean_abs / math.sqrt(self.Qp)

    @torch.no_grad()
    def init_from(self, x):
        """
        Initialize signedness and step size from one tensor
        """
        if self.signed_mode == 'auto' and not self.is_weight:
            self.signed = 1 if float(x.min()) < 0 else 0
        self._set_levels()

        x = x.detach()
        if self.per_channel:
            flat = x.flatten(1)
            self.s.data.copy_(self._step_from_stats(flat.abs().mean(dim=1),
                                                    flat.abs().amax(dim=1)))
        else:
            self.s.data.fill_(self._step_from_stats(x.abs().mean().item(),
                                                    x.abs().max().item()))
        self._ready = True

    def forward(self, x):
        if not self._ready:
            self.init_from(x)

        n_elem = x.numel() if (self.is_weight and not self.per_channel) else x[0].numel()
        g = 1.0 / math.sqrt(n_elem * self.Qp)

        s = grad_scale(self.s.abs().clamp(min=1e-8), g)
        if self.per_channel:
            s = s.view(-1, *([1] * (x.dim() - 1)))

        u = (x / s).clamp(self.Qn, self.Qp)
        if self.cache_round:
            self.round_cache = (u.detach(), s.detach())
        if self.cache_u:
            self.u_cache = (x / s.detach()).clamp(self.Qn, self.Qp)
        return round_pass(u) * s
