'''
EWGS: Network Quantization with Element-wise Gradient Scaling
Lee et al., CVPR 2021 -- https://arxiv.org/abs/2104.00903

A drop-in replacement for the straight-through estimator. The STE passes the
gradient through unchanged, ignoring how far the latent value sits from the
grid point it rounded to. EWGS scales each element by

    scale = 1 + delta * sign(g) * (u - round(u))

so a coordinate whose gradient pushes it away from its grid point gets a larger
gradient, and one being pushed back onto it gets a smaller one.

The reference normalizes its input to [0, 1], where the rounding error spans
+-0.5/(2^b - 1). Here it is in grid units, spanning +-0.5 at every bit width,
so delta means the same thing across bit widths; their fixed delta=0.001
coincides with ours only at 1 bit. delta <= 2 keeps the scale non-negative.

This is the one place a custom autograd Function is needed: the scale depends
on the incoming gradient, which the detach trick cannot express.
'''
import torch


class _EWGS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, delta):
        xq = x.round()
        ctx.save_for_backward(x - xq)
        ctx.delta = delta
        return xq

    @staticmethod
    def backward(ctx, g):
        diff, = ctx.saved_tensors
        return g * (1 + ctx.delta * g.sign() * diff), None


def ewgs_pass(x, delta):
    """Forward: round(x). Backward: the STE scaled element-wise."""
    assert 0.0 <= delta <= 2.0, delta
    return _EWGS.apply(x, delta)
