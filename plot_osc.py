"""Figures for the oscillation probe.

    python plot_osc.py runs/lsq.npz runs/sam.npz -o figs

Works with a single run too; trainer.py calls make_figures() directly so that
separately launched runs each produce their own pair of figures.
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['0.45', '#d1495b', '#2e6f95', '#7a9e3f']
PANELS = [('freq', 'oscillation frequency', 'reversals / step'),
          ('amp_w_rel', 'amplitude (weight space)', r'$\sqrt{\sum s^2\mathrm{Var}[n]}\,/\,\|Q(w)\|$'),
          ('amp_l', 'amplitude (loss space)', r'$\sqrt{\sum \mathrm{gain}^2\mathrm{Var}[n]}$  [nats]')]

parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='+', metavar='FILE[:LABEL]')
parser.add_argument('-o', '--out', default='figs')
parser.add_argument('--n-show', default=5, type=int, metavar='K',
                    help='top-k oscillating weights to trace (default: 5)')
parser.add_argument('--trace-steps', default=0, type=int,
                    help='steps to display; 0 shows the whole recorded window')
parser.add_argument('--pick', default='base', choices=['base', 'union', 'each'],
                    help='rank trace weights by the baseline run, by the max over runs, '
                         'or within each run separately')
parser.add_argument('--logy', action='store_true', help='log scale on the amplitude panels')


def load(spec):
    path, _, label = spec.partition(':')
    d = np.load(path, allow_pickle=True)
    return label or str(d['label']) or os.path.splitext(os.path.basename(path))[0], d


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=.25, lw=.6)
    ax.set_axisbelow(True)


def reversals(tr):
    return (np.diff(np.sign(np.diff(tr, axis=0)), axis=0) != 0).sum(0)


def window(d, args):
    return d['traces'][-args.trace_steps:] if args.trace_steps else d['traces']


def span(d):
    n = len(d['traces'])
    e = float(d['epoch_len']) if 'epoch_len' in d else 0
    return '%d steps' % n + (' (%.1f epochs)' % (n / e) if e else '')


def fig_traces(runs, args):
    rev = [reversals(d['traces']) for _, d in runs]
    if args.pick == 'each':
        picks = [np.argsort(-r)[:args.n_show] for r in rev]
    else:
        rank = rev[0] if args.pick == 'base' else np.max(rev, axis=0)
        picks = [np.argsort(-rank)[:args.n_show]] * len(runs)

    fig, axes = plt.subplots(args.n_show, len(runs), sharex=True,
                             figsize=(3.1 * len(runs), 0.72 * args.n_show + 1.2),
                             squeeze=False)
    for c, (label, d) in enumerate(runs):
        tr = window(d, args)
        for r, j in enumerate(picks[c]):
            ax = axes[r][c]
            ax.step(np.arange(len(tr)), tr[:, j], where='post',
                    color=COLORS[c % len(COLORS)], lw=1.0)
            ax.set_yticks(np.unique(tr[:, j]))
            ax.tick_params(labelsize=7, length=2)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if c == 0 or args.pick == 'each':
                ax.set_ylabel('w%d' % j, fontsize=7, rotation=0,
                              ha='right', va='center')
            if r == 0:
                ax.set_title(label, fontsize=10)
        axes[-1][c].set_xlabel('training step', fontsize=8)

    how = ('ranked over the last ' + span(runs[0][1]) if len(runs) == 1 else
           {'base': 'ranked under ' + runs[0][0],
            'union': 'ranked by the max over runs',
            'each': 'ranked within each run'}[args.pick]
           + ' over the last ' + span(runs[0][1]))
    fig.suptitle('top-%d oscillating weights: rounded level $n_i$\n%s'
                 % (args.n_show, how), fontsize=9, y=.995)
    fig.tight_layout(rect=[0, 0, 1, .93])
    return fig


def fig_summary(runs, args):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6))

    for ax, (key, title, ylab) in zip(axes.flat, PANELS):
        for c, (label, d) in enumerate(runs):
            y = d[key]
            ax.plot(np.arange(1, len(y) + 1), y, color=COLORS[c % len(COLORS)],
                    lw=1.6, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('epoch', fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=8)
        if args.logy and key != 'freq':
            ax.set_yscale('log')
        style(ax)

        base = runs[0][1][key]
        tail = max(1, len(base) // 10)
        b = base[-tail:].mean()
        txt = []
        for label, d in runs[1:]:
            v = d[key][-tail:].mean()
            txt.append('%s  %.3g $\\rightarrow$ %.3g  (%+.0f%%)'
                       % (label, b, v, 100 * (v / b - 1) if b else 0))
        if txt:
            ax.text(.97, .95, '\n'.join(txt), transform=ax.transAxes, fontsize=7.5,
                    ha='right', va='top',
                    bbox=dict(fc='white', ec='0.8', lw=.6, alpha=.9, pad=3))

    ax = axes.flat[3]
    for c, (label, d) in enumerate(runs):
        h, (bins, vmax) = d['hist_var'], d['bins']
        edges = np.linspace(0, vmax, int(bins) + 1)
        ccdf = h[::-1].cumsum()[::-1] / d['n_weights']
        ax.semilogy(edges[:-1], np.maximum(ccdf, 1e-8),
                    color=COLORS[c % len(COLORS)], lw=1.6, label=label)
    ax.set_title('per-weight amplitude, tail', fontsize=10)
    ax.set_xlabel(r'$\mathrm{Var}[n_i]$', fontsize=8)
    ax.set_ylabel(r'fraction of weights $>x$', fontsize=8)
    ax.set_ylim(1e-6, 1.2)
    ax.tick_params(labelsize=8)
    style(ax)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def fig_grid(runs, args):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for c, (label, d) in enumerate(runs):
        h = d['hist_grid']
        tail = max(1, len(h) // 10)
        p = h[-tail:].mean(0)
        x = (np.arange(len(p)) + .5) * 0.5 / len(p)
        u = 1.0 / len(p)
        axes[0].plot(x, p / u, color=COLORS[c % len(COLORS)], lw=1.6, label=label)
        axes[1].plot(x, 100 * (p / u - 1), color=COLORS[c % len(COLORS)], lw=1.6,
                     label=label)

    for ax in axes:
        ax.axvline(0, color='0.7', lw=.8, ls=':')
        ax.set_xlabel(r'$|u_i - \mathrm{round}(u_i)|$   (0 = grid point,'
                      '  0.5 = boundary)', fontsize=8)
        ax.tick_params(labelsize=8)
        style(ax)
    axes[0].axhline(1, color='0.6', lw=.9, ls='--')
    axes[0].set_ylabel('density / uniform', fontsize=8)
    axes[0].set_title('distance to the nearest grid point', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].axhline(0, color='0.6', lw=.9, ls='--')
    axes[1].set_ylabel('excess over uniform  [%]', fontsize=8)
    axes[1].set_title('same, relative to uniform', fontsize=10)

    fig.suptitle('clipped weights excluded; averaged over the last %d epochs'
                 % max(1, len(runs[0][1]['hist_grid']) // 10), fontsize=8, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95])
    return fig


def make_figures(runs, out, prefix='', n_show=5, trace_steps=0, pick='base',
                 logy=False):
    args = SimpleNamespace(n_show=n_show, trace_steps=trace_steps, pick=pick,
                           logy=logy)
    os.makedirs(out, exist_ok=True)
    figs = [('traces', fig_traces(runs, args)), ('summary', fig_summary(runs, args))]
    if all('hist_grid' in d for _, d in runs):
        figs.append(('grid', fig_grid(runs, args)))
    paths = []
    for name, fig in figs:
        stem = os.path.join(out, '%s%s' % (prefix, name))
        for ext in ('pdf', 'png'):
            fig.savefig('%s.%s' % (stem, ext), dpi=200)
        plt.close(fig)
        paths.append(stem + '.pdf')
    return paths


def main():
    args = parser.parse_args()
    runs = [load(s) for s in args.runs]
    prefix = '' if len(runs) > 1 else runs[0][0].replace(os.sep, '_') + '_'
    for p in make_figures(runs, args.out, prefix, args.n_show, args.trace_steps,
                          args.pick, args.logy):
        print('=> %s' % p)


if __name__ == '__main__':
    main()
