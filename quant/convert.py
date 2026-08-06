import torch.nn as nn

from .lsq import LsqQuantizer
from .modules import QConv2d, QLinear

def convert(model, w_bits=4, a_bits=4, first_last_bits=8,
            per_channel=True, a_signed='auto', init_mode='lsq', ewgs=0.0):
    """
    Replace every Conv2d/Linear with its quantized counterpart, in place.
    """
    def parent_of(model, name):
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    def is_first_last(idx, n_layers):
        return idx == 0 or idx == n_layers - 1

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))
              and not isinstance(m, (QConv2d, QLinear))]
    n = len(layers)
    assert n > 0, 'nothing to convert (already converted?)'

    for idx, (name, mod) in enumerate(layers):
        wb, ab = w_bits, a_bits
        if is_first_last(idx, n):
            if first_last_bits == 0:
                continue                     # leave in FP32
            wb = first_last_bits if w_bits else 0
            ab = first_last_bits if a_bits else 0

        wq = None
        if wb:
            wq = LsqQuantizer(wb, is_weight=True, per_channel=per_channel,
                              num_channels=mod.weight.shape[0] if per_channel else None,
                              init_mode=init_mode)
            wq.init_from(mod.weight)
            wq.ewgs = ewgs
        aq = (LsqQuantizer(ab, is_weight=False, signed_mode=a_signed,
                           init_mode=init_mode) if ab else None)
        if aq is not None:
            aq.ewgs = ewgs

        cls = QConv2d if isinstance(mod, nn.Conv2d) else QLinear
        parent, attr = parent_of(model, name)
        setattr(parent, attr, cls.from_float(mod, wq, aq))

    return model


def quant_param_groups(model, args):
    """
    weights as usual, step sizes with NO weight decay.
    """
    step_ids = {id(p) for m in model.modules() if isinstance(m, LsqQuantizer)
                for p in m.parameters(recurse=False)}
    step, other = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (step if id(p) in step_ids else other).append(p)

    return [
        {'params': other, 'weight_decay': args.weight_decay,
         'lr': args.lr, 'name': 'weights'},
        {'params': step, 'weight_decay': 0.0,
         'lr': args.lr * args.q_step_lr_scale, 'name': 'step_size'},
    ]
