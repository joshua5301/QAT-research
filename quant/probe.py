import torch

from .modules import QConv2d, QLinear


class OscProbe:
    """Oscillation frequency and amplitude of the rounded levels.

    freq   reversal rate: flips that undo the previous flip
    var    Var[n] over a short window; equals p(1-p) for a two-level swing
    amp_w  sqrt(sum s^2 Var[n]), the wobble of Q(w); amp_w_rel divides by ||Q(w)||
    amp_l  sqrt(sum gain^2 Var[n]), the oscillation-induced loss std, in nats
    margin |r - 0.5|, distance to the rounding boundary in units of s
    tight  fraction with margin < the threshold
    drift  |fast EMA - slow EMA| of n, to tell drift from oscillation
    multi  fraction with Var[n] > 0.25, i.e. swings wider than one level
    """

    KEYS = ('freq', 'var', 'margin', 'tight', 'drift', 'multi')
    ALL_KEYS = KEYS + ('amp_w', 'amp_w_rel', 'amp_l')

    def __init__(self, model, momentum=0.01, slow_momentum=0.001, tight=0.05,
                 trace_steps=10000, trace_width=1024, trace_layer=None):
        self.mom = momentum
        self.slow_mom = slow_momentum
        self.tight = tight

        self.layers = [m for m in model.modules()
                       if isinstance(m, (QConv2d, QLinear)) and m.wq is not None]
        assert self.layers, 'OscProbe needs quantized weights (--qat)'
        for m in self.layers:
            m.wq.cache_round = True
        self.state = [None] * len(self.layers)

        self.history = []
        self._acc, self._n, self.epoch_len = dict.fromkeys(self.ALL_KEYS, 0.0), 0, 0
        self.trace_layer = len(self.layers) // 2 if trace_layer is None else trace_layer
        self.trace_steps, self.trace_width = trace_steps, trace_width
        self.traces, self.trace_idx, self.trace_pos = None, None, 0

    @torch.no_grad()
    def step(self):
        """Call after the clean backward, before optimizer.step()."""
        assert self.layers[0].weight.grad is not None, \
            'step() needs a backward pass first'
        rows, total = [], 0

        for i, m in enumerate(self.layers):
            u, s = m.wq.round_cache
            n = u.round()

            st = self.state[i]
            if st is None:
                st = self.state[i] = dict(
                    n=n.clone(), dir=torch.zeros_like(n), f=torch.zeros_like(n),
                    mu=n.clone(), nu=n.square(), slow=n.clone())

            dn = (n - st['n']).sign()
            moved = dn != 0
            rev = (moved & (dn == -st['dir'])).to(n.dtype)
            st['dir'] = torch.where(moved, dn, st['dir'])
            st['n'] = n

            st['f'] += self.mom * (rev - st['f'])
            st['mu'] += self.mom * (n - st['mu'])
            st['nu'] += self.mom * (n.square() - st['nu'])
            st['slow'] += self.slow_mom * (n - st['slow'])

            # STE makes weight.grad equal dL/dQ(w) inside the clamp range
            var = (st['nu'] - st['mu'].square()).clamp(min=0)
            gain = (m.weight.grad * s).abs()
            margin = (u - u.floor() - 0.5).abs()

            rows.append(torch.stack([
                st['f'].sum(),
                var.sum(),
                margin.sum(),
                (margin < self.tight).to(n.dtype).sum(),
                (st['mu'] - st['slow']).abs().sum(),
                (var > 0.25).to(n.dtype).sum(),
                (s.square() * var).sum(),
                (gain.square() * var).sum(),
                (n * s).square().sum(),
            ]))
            total += n.numel()

            if i == self.trace_layer:
                self._trace(n)

        *means, aw, al, qq = torch.stack(rows).sum(0).tolist()
        out = {k: v / total for k, v in zip(self.KEYS, means)}
        out['amp_w'] = aw ** 0.5
        out['amp_w_rel'] = (aw / qq) ** 0.5 if qq > 0 else 0.0
        out['amp_l'] = al ** 0.5

        for k in self.ALL_KEYS:
            self._acc[k] += out[k]
        self._n += 1
        return out

    def _trace(self, n):
        flat = n.flatten()
        if self.traces is None:
            k = min(self.trace_width, flat.numel())
            self.trace_idx = torch.linspace(0, flat.numel() - 1, k).long().to(flat.device)
            self.traces = torch.zeros(self.trace_steps, k, dtype=torch.int8,
                                      device=flat.device)
        self.traces[self.trace_pos % self.trace_steps] = flat[self.trace_idx].to(torch.int8)
        self.trace_pos += 1

    def snapshot(self):
        if self._n:
            self.history.append({k: v / self._n for k, v in self._acc.items()})
            self._acc, self._n, self.epoch_len = \
                dict.fromkeys(self.ALL_KEYS, 0.0), 0, self._n

    @torch.no_grad()
    def save(self, path, label='', bins=128, vmax=0.5):
        import numpy as np

        self.snapshot()
        f = torch.cat([st['f'].flatten() for st in self.state])
        var = torch.cat([(st['nu'] - st['mu'].square()).clamp(min=0).flatten()
                         for st in self.state])

        p, t = self.trace_pos, self.trace_steps
        tr = self.traces[:p] if p < t else torch.roll(self.traces, -p % t, 0)

        np.savez(path, label=label, bins=np.float32([bins, vmax]),
                 epoch_len=float(self.epoch_len),
                 hist_f=torch.histc(f, bins, 0, vmax).cpu().numpy(),
                 hist_var=torch.histc(var, bins, 0, vmax).cpu().numpy(),
                 n_weights=float(f.numel()),
                 traces=tr.cpu().numpy(),
                 **{k: np.float32([h[k] for h in self.history])
                    for k in self.ALL_KEYS})

    @staticmethod
    def format(st):
        return ('osc: freq {freq:.4f} var {var:.4f} | ampW {amp_w:.3f} '
                '({amp_w_rel:.4f}) ampL {amp_l:.4f} | margin {margin:.3f} '
                'tight {tight:.4f} | drift {drift:.4f} multi {multi:.5f}'
                .format(**st))
