'''
The continuous parameters a SAM ball may include, shared by SAQ and
DiscreteSAM so the two read --*-cont the same way.
'''
import torch.nn as nn

from ..lsq import LsqQuantizer
from ..modules import QConv2d, QLinear

CONT = ('bn', 'bias', 'wscale', 'ascale')
DEFAULT = ('bn', 'bias')


def continuous_params(model, cont):
    """Parameters of the selected groups, in module order.

    wscale/ascale are the LSQ step sizes; perturbing those moves the
    quantization grid rather than the weights on it, which is why they are
    off by default.
    """
    cont = set(cont)
    assert cont <= set(CONT), sorted(cont - set(CONT))
    out = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            key = 'bn'
        elif isinstance(m, LsqQuantizer):
            key = 'wscale' if m.is_weight else 'ascale'
        elif isinstance(m, (QConv2d, QLinear)):
            if 'bias' in cont and m.bias is not None:
                out.append(m.bias)
            continue
        else:
            continue
        if key in cont:
            out += [p for p in m.parameters(recurse=False) if p.requires_grad]
    return out
