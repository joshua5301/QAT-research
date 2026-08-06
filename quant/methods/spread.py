'''
The per-coordinate rounding spread, shared by the penalty and by the ball it
shapes.
'''
import math


def spread(m):
    """s * |sin(pi u)|, weight-shaped. Needs m.wq.cache_u.

    How far rounding moves coordinate i, in weight units: zero once the latent
    weight sits on a grid point, largest on a decision boundary, and free of
    round(). Differentiable through u, since u_cache holds w / s.detach()
    before the rounding.
    """
    u, s = m.wq.u_cache
    return s * (math.pi * u).sin().abs()
