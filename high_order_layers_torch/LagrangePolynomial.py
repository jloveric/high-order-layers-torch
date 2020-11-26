import numpy as np
import math
import torch
from .Basis import *


def chebyshevLobatto(n: int):
    """
    Compute the chebyshev lobatto points which
    are in the range [-1.0, 1.0]
    Args :
        n : number of points
    Returns :
        A tensor of length n with x locations from
        negative to positive including -1 and 1
         [-1,...,+1]
    """
    k = torch.arange(0, n)

    ans = -torch.cos(k * math.pi / (n - 1))

    ans = torch.where(torch.abs(ans) < 1e-15, 0*ans, ans)

    return ans


class FourierBasis:
    def __init__(self, length: float):
        self.length = length

    def __call__(self, x, j: int):

        if j == 0:
            return 0.5+0.0*x

        i = (j+1)//2
        if j % 2 == 0:
            ans = torch.cos(math.pi*i*x/self.length)
        else:
            ans = torch.sin(math.pi*i*x/self.length)
        return ans


class LagrangeBasis:
    def __init__(self, n: int, length: float = 2.0):
        self.n = n
        self.X = (length/2.0)*chebyshevLobatto(n)

    def __call__(self, x, j: int):

        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        b = torch.stack(b)
        ans = torch.prod(b, dim=0)
        return ans


class LagrangeExpand(BasisExpand):
    def __init__(self, n: int):
        super().__init__(LagrangeBasis(n), n)


class PiecewisePolynomialExpand(PiecewiseExpand):
    def __init__(self, n: int, segments: int):
        super().__init__(basis=LagrangeBasis(n), n=n, segments=segments)


class PiecewiseDiscontinuousPolynomialExpand(PiecewiseDiscontinuousExpand):
    def __init__(self, n: int, segments: int):
        super().__init__(basis=LagrangeBasis(n), n=n, segments=segments)


class FourierExpand(BasisExpand):
    def __init__(self, n: int):
        super().__init__(FourierBasis(length=1), n)


class LagrangePolyFlat(BasisFlat):
    def __init__(self, n: int, length: float = 2.0):
        super().__init__(n, LagrangeBasis(n))


class FourierSeriesFlat(BasisFlat):
    def __init__(self, n: int, length: int = 1.0):
        super().__init__(n, FourierBasis(length))


# This may be redundant.
class LagrangePoly(Basis):
    def __init__(self, n: int):
        super().__init__(n=n, basis=LagrangeBasis(n))
