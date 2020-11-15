import numpy as np
import math
import torch
from .Basis import *


def chebyshevLobatto(n):

    k = torch.arange(0, n)

    ans = -torch.cos(k * math.pi / (n - 1))

    ans = torch.where(torch.abs(ans) < 1e-15, 0*ans, ans)

    return ans


class FourierBasis:
    def __init__(self, length):
        self.length = length

    def __call__(self, x, j):

        if j == 0:
            return 0.5+0.0*x

        i = (j+1)//2
        if j % 2 == 0:
            ans = torch.cos(math.pi*i*x/self.length)
        else:
            ans = torch.sin(math.pi*i*x/self.length)
        return ans


class LagrangeBasis:
    def __init__(self, n):
        self.n = n
        self.X = chebyshevLobatto(n)

    def __call__(self, x, j):

        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        b = torch.stack(b)
        ans = torch.prod(b, dim=0)
        return ans


class LagrangeExpand(BasisExpand):
    def __init__(self, n):
        super().__init__(LagrangeBasis(n), n)


class FourierExpand(BasisExpand):
    def __init__(self, n):
        super().__init__(FourierBasis(length=1), n)


class LagrangePolyFlat(BasisFlat):
    def __init__(self, n):
        super().__init__(n, LagrangeBasis(n))


class FourierSeriesFlat(BasisFlat):
    def __init__(self, n, length=1.0):
        super().__init__(n, FourierBasis(length))


# This may be redundant.
class LagrangePoly(Basis):
    def __init__(self, n):
        super().__init__(n=n, basis=LagrangeBasis(n))
