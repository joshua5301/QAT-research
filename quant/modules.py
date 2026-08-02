import torch.nn as nn
import torch.nn.functional as F

class QConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super(QConv2d, self).__init__(*args, **kwargs)
        self.wq = None
        self.aq = None

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
        return self._conv_forward(x, w, self.bias)


class QLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super(QLinear, self).__init__(*args, **kwargs)
        self.wq = None
        self.aq = None

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
        return F.linear(x, w, self.bias)
