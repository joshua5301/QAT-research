'''
Where the SAM ascent lands: per-layer flip counts.
'''
import torch

from .modules import QConv2d, QLinear


class FlipProbe:
    """Counts the coordinates DiscreteSAM flips, per layer.

    Averaged over the steps since the last snapshot. `frac` is relative to the
    layer's own weight count, `share` to all flips in the step, so a layer can
    be a large share of the ascent while barely being perturbed itself.
    """

    def __init__(self, model):
        self.layers = [(n, m) for n, m in model.named_modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'FlipProbe needs quantized weights (--qat)'
        self.numel = [m.weight.numel() for _, m in self.layers]
        self.flips = torch.zeros(len(self.layers))
        self.steps = 0
        self.history = []

    @torch.no_grad()
    def step(self):
        """Call after ascent_step(), before the second backward."""
        for i, (_, m) in enumerate(self.layers):
            if m.eps is not None:
                self.flips[i] += float((m.eps != 0).sum())
        self.steps += 1

    def snapshot(self):
        if self.steps:
            self.history.append(self.flips / self.steps)
            self.flips = torch.zeros(len(self.layers))
            self.steps = 0

    def report(self):
        """One line per layer, from the last snapshot."""
        if not self.history:
            return 'flips: no steps recorded'
        f = self.history[-1]
        total = float(f.sum())
        out = ['flips: {:.0f} per step, {:.4f}% of all weights'.format(
            total, 100.0 * total / sum(self.numel))]
        out.append('     {:<3s} {:<26s} {:>9s} {:>8s} {:>8s} {:>8s}'.format(
            'idx', 'layer', 'weights', 'flips', 'frac%', 'share%'))
        for i, ((name, _), n) in enumerate(zip(self.layers, self.numel)):
            out.append('     {:<3d} {:<26s} {:>9d} {:>8.1f} {:>8.3f} {:>8.2f}'.format(
                i, name[:26], n, float(f[i]), 100.0 * float(f[i]) / n,
                100.0 * float(f[i]) / total if total else 0.0))
        return '\n'.join(out)

    def save(self, path, label=''):
        import os

        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.snapshot()
        np.savez(path, label=label,
                 names=np.array([n for n, _ in self.layers]),
                 numel=np.int64(self.numel),
                 flips=np.float32([h.tolist() for h in self.history]))
