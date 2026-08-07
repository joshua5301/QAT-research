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


def spread_cont(p):
    """pi * |w|, the Delta -> infinity limit of spread().

    An unquantized parameter is one whose grid spacing is infinite, and
    s |sin(pi w / s)| -> pi |w| there. So the continuous block is not a special
    case bolted on beside the quantized one -- it is the same operator at the
    limit, and it happens to be ASAM's metric. Also required on dimensional
    grounds: (m * g)^2 carries weight^2 * gradient^2, so a dimensionless T = I
    could not share a norm with it.
    """
    return math.pi * p.abs()
