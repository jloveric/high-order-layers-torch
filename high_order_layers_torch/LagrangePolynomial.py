import math

import torch
from torch import Tensor

from .Basis import *


def chebyshevLobatto(n: int):
    """
    Compute the chebyshev lobatto points which
    are in the range [-1.0, 1.0]
    :param n: number of points
    :returns :
        A tensor of length n with x locations from
        negative to positive including -1 and 1
         [-1,...,+1]
    """
    if n == 1:
        return torch.tensor([0.0])

    return -torch.cos(torch.pi * torch.arange(n) / (n - 1))


class FourierBasis:
    def __init__(self, length: float):
        """
        Fourier basis functions [sin, cos]
        Args :
            length : the length of the basis function. A value
            of 1 means there is periodicity 1
        """
        self.length = length
        self.num_basis = None # Apparently defined elsewhere? How does this work!

    def __call__(self, x: Tensor, j: int):
        """
        Compute the value at x for the given component
        of the fourier basis function.
        Args :
            x : the point of interest (can be of any shape)
        """
        if j == 0:
            return 0.5 + 0.0 * x

        i = (j + 1) // 2
        if j % 2 == 0:
            ans = torch.cos(2.0 * math.pi * i * x / self.length)
        else:
            ans = torch.sin(2.0 * math.pi * i * x / self.length)
        return ans


class LagrangeBasis:
    def __init__(self, n: int, length: float = 2.0):
        self.n = n
        self.X = (length / 2.0) * chebyshevLobatto(n)
        self.denominators = self._compute_denominators()
        self.num_basis = n

    def _compute_denominators(self):
        denom = torch.ones((self.n, self.n), dtype=torch.float32)
        for j in range(self.n):
            for m in range(self.n):
                if m != j:
                    denom[j, m] = self.X[j] - self.X[m]
        return denom

    def __call__(self, x, j: int):
        x_diff = x.unsqueeze(-1) - self.X  # Ensure broadcasting
        b = torch.where(
            torch.arange(self.n) != j, x_diff / self.denominators[j], torch.tensor(1.0)
        )
        ans = torch.prod(b, dim=-1)
        return ans


class LagrangeBasisND:

    def __init__(self, n: int, length: float = 2.0, dimensions: int = 2):
        self.n = n
        self.dimensions = dimensions
        self.X = (length / 2.0) * chebyshevLobatto(n)
        self.denominators = self._compute_denominators()
        self.num_basis = int(math.pow(n, dimensions))

    def _compute_denominators(self):
        denom = torch.ones([self.n, self.n], dtype=torch.float32)

        for j in range(self.n):
            for m in range(self.n):
                if m != j:
                    denom[j, m] = self.X[j] - self.X[m]
        return denom

    def __call__(self, x, index: list[int]):
        """
        :param x: [batch, inputs, dimensions]
        :param index : [dimensions]
        :returns: basis value [batch, inputs]
        """
        x_diff = x.unsqueeze(-1) - self.X  # [batch, inputs, dimensions, basis]
        r = 1.0
        for i, basis_i in enumerate(index):
            b = torch.where(
                torch.arange(self.n) != basis_i,
                x_diff[:, :, i, :] / self.denominators[basis_i],
                torch.tensor(1.0),
            )
            r *= torch.prod(b, dim=-1)

        return r


class LagrangeBasis1:
    """
    TODO: Degenerate case, test this and see if it works with everything else.

    """

    def __init__(self, length: float = 2.0):
        self.n = 1
        self.X = torch.tensor([0.0])
        self.num_basis=1

    def __call__(self, x, j: int):
        b = torch.ones_like(x)
        return b


def get_lagrange_basis(n: int, length: float = 2.0):
    if n == 1:
        return LagrangeBasis1(length=length)
    else:
        return LagrangeBasis(n, length=length)


class LagrangeExpand(BasisExpand):
    def __init__(self, n: int, length: float = 2.0):
        super().__init__(get_lagrange_basis(n, length), n)


class PiecewisePolynomialExpand(PiecewiseExpand):
    def __init__(self, n: int, segments: int, length: float = 2.0):
        super().__init__(
            basis=get_lagrange_basis(n, length), n=n, segments=segments, length=length
        )


class PiecewisePolynomialExpand1d(PiecewiseExpand1d):
    def __init__(self, n: int, segments: int, length: float = 2.0):
        super().__init__(
            basis=get_lagrange_basis(n, length), n=n, segments=segments, length=length
        )


class PiecewiseDiscontinuousPolynomialExpand(PiecewiseDiscontinuousExpand):
    def __init__(self, n: int, segments: int, length: float = 2.0):
        super().__init__(
            basis=get_lagrange_basis(n, length), n=n, segments=segments, length=length
        )


class PiecewiseDiscontinuousPolynomialExpand1d(PiecewiseDiscontinuousExpand1d):
    def __init__(self, n: int, segments: int, length: float = 2.0):
        super().__init__(
            basis=get_lagrange_basis(n, length), n=n, segments=segments, length=length
        )


class FourierExpand(BasisExpand):
    def __init__(self, n: int, length: float):
        super().__init__(FourierBasis(length=length), n)


class LagrangePolyFlat(BasisFlat):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, get_lagrange_basis(n, length), **kwargs)


class LagrangePolyFlatND(BasisFlatND):
    def __init__(self, n: int, length: float = 2.0, dimensions: int = 2, **kwargs):
        super().__init__(
            n,
            LagrangeBasisND(n, length, dimensions=dimensions),
            dimensions=dimensions,
            **kwargs
        )


class LagrangePolyFlatProd(BasisFlatProd):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, get_lagrange_basis(n, length), **kwargs)


class LagrangePoly(Basis):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, get_lagrange_basis(n, length=length), **kwargs)


class LagrangePolyProd(BasisProd):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, get_lagrange_basis(n, length), **kwargs)


class FourierSeriesFlat(BasisFlat):
    def __init__(self, n: int, length: int = 1.0, **kwargs):
        super().__init__(n, FourierBasis(length), **kwargs)
