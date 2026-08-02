import torch.nn as nn
import torch.nn.functional as F


class _SamHook:
    """Retain the grad of Q(w) on pass 1, add the flip epsilon on pass 2."""

    def _sam_hook(self, w):
        if not self.sam:
            return w
        if w.requires_grad:
            w.retain_grad()
            self.wq_out = w
        return w if self.eps is None else w + self.eps


class QConv2d(_SamHook, nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super(QConv2d, self).__init__(*args, **kwargs)
        self.wq = None
        self.aq = None
        self.sam = False        # GridSAM hooks below; plain attrs, not state
        self.eps = None         # added after quantization on the 2nd pass
        self.wq_out = None      # Q(w) with a retained grad from the 1st pass

    @classmethod
    def from_float(cls, m, wq, aq):
        q = cls(m.in_channels, m.out_channels, m.kernel_size, m.stride,
                m.padding, m.dilation, m.groups, bias=m.bias is not None,
                padding_mode=m.padding_mode)
        q.weight = m.weight
        q.bias = m.bias
        q.wq, q.aq = wq, aq
        return q

    def forward(self, x):
        if self.aq is not None:
            x = self.aq(x)
        w = self.weight if self.wq is None else self.wq(self.weight)
        w = self._sam_hook(w)
        return self._conv_forward(x, w, self.bias)


class QLinear(_SamHook, nn.Linear):

    def __init__(self, *args, **kwargs):
        super(QLinear, self).__init__(*args, **kwargs)
        self.wq = None
        self.aq = None
        self.sam = False
        self.eps = None
        self.wq_out = None

    @classmethod
    def from_float(cls, m, wq, aq):
        q = cls(m.in_features, m.out_features, bias=m.bias is not None)
        q.weight = m.weight
        q.bias = m.bias
        q.wq, q.aq = wq, aq
        return q

    def forward(self, x):
        if self.aq is not None:
            x = self.aq(x)
        w = self.weight if self.wq is None else self.wq(self.weight)
        w = self._sam_hook(w)
        return F.linear(x, w, self.bias)
