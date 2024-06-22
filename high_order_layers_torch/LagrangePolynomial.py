import math
from typing import List, Union
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


# class LagrangeBasisND:
#     """
#     Single N dimensional element with Lagrange basis interpolation.
#     """
#     def __init__(
#         self, n: int, length: float = 2.0, dimensions: int = 2, device: str = "cpu", **kwargs
#     ):
#         self.n = n
#         self.dimensions = dimensions
#         self.X = (length / 2.0) * chebyshevLobatto(n).to(device)
#         self.device = device
#         self.denominators = self._compute_denominators()
#         self.num_basis = int(math.pow(n, dimensions))
        
#         a = torch.arange(n)
#         self.indexes = (
#             torch.stack(torch.meshgrid([a] * dimensions, indexing="ij"))
#             .reshape(dimensions, -1)
#             .T.long().to(self.device)
#         )

#     def _compute_denominators(self):
#         X_diff = self.X.unsqueeze(0) - self.X.unsqueeze(1)  # [n, n]
#         denom = torch.where(X_diff == 0, torch.tensor(1.0, device=self.device), X_diff)
#         return denom

#     def _compute_basis(self, x, indexes):
#         """
#         Computes the basis values for all index combinations.
#         :param x: [batch, inputs, dimensions]
#         :param indexes: [num_basis, dimensions]
#         :returns: basis values [num_basis, batch, inputs]
#         """
#         x_diff = x.unsqueeze(-1) - self.X  # [batch, inputs, dimensions, n]
#         mask = (indexes.unsqueeze(1).unsqueeze(2).unsqueeze(4) != torch.arange(self.n, device=self.device).view(1, 1, 1, 1, self.n))
#         denominators = self.denominators[indexes]  # [num_basis, dimensions, n]

#         b = torch.where(mask, x_diff.unsqueeze(0) / denominators.unsqueeze(1).unsqueeze(2), torch.tensor(1.0, device=self.device))
#         #print('b.shape', b.shape)
#         r = torch.prod(b, dim=-1)  # [num_basis, batch, inputs, dimensions]

#         return r.prod(dim=-1)  # [num_basis, batch, inputs]

#     def interpolate(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
#         """
#         Interpolates the input using the Lagrange basis.
#         :param x: size[batch, inputs, dimensions]
#         :param w: size[output, inputs, num_basis]
#         :returns: size[batch, output]
#         """
#         basis = self._compute_basis(x, self.indexes)  # [num_basis, batch, inputs]
#         #print('bassis.shape', basis.shape, 'w.shape', w.shape)
#         out_sum = torch.einsum("ibk,oki->bo", basis, w)  # [batch, output]

#         return out_sum


import torch
import math
from typing import List

class LagrangeBasisND:
    """
    Single N dimensional element with Lagrange basis interpolation.
    Supports different n values for each dimension.
    """
    def __init__(
        self, n: Union[List[int],int], length: float = 2.0, device: str = "cpu", **kwargs
    ):
        self.n = n
        self.dimensions = len(n)
        self.X = [(length / 2.0) * chebyshevLobatto(ni).to(device) for ni in n]
        self.device = device
        self.denominators = self._compute_denominators()
        self.num_basis = math.prod(n)
        
        self.indexes = self._compute_indexes()

    def _compute_denominators(self):
        return [
            torch.where(X_diff == 0, torch.tensor(1.0, device=self.device), X_diff)
            for X_diff in [Xi.unsqueeze(0) - Xi.unsqueeze(1) for Xi in self.X]
        ]

    def _compute_indexes(self):
        ranges = [torch.arange(ni) for ni in self.n]
        meshgrid = torch.stack(torch.meshgrid(ranges, indexing="ij"))
        return meshgrid.reshape(self.dimensions, -1).T.long().to(self.device)

    def _compute_basis(self, x, indexes):
        """
        Computes the basis values for all index combinations.
        :param x: [batch, inputs, dimensions]
        :param indexes: [num_basis, dimensions]
        :returns: basis values [num_basis, batch, inputs]
        """
        b_list = []
        for d in range(self.dimensions):
            x_diff = x[..., d].unsqueeze(-1) - self.X[d]  # [batch, inputs, n[d]]
            mask = (indexes[:, d].unsqueeze(1).unsqueeze(2) != torch.arange(self.n[d], device=self.device))
            denominators = self.denominators[d][indexes[:, d]]  # [num_basis, n[d]]

            # Reshape x_diff and denominators for proper broadcasting
            x_diff_expanded = x_diff.unsqueeze(0)  # [1, batch, inputs, n[d]]
            denominators_expanded = denominators.unsqueeze(1).unsqueeze(2)  # [num_basis, 1, 1, n[d]]

            # Ensure mask has the correct shape
            mask = mask.unsqueeze(1)  # [num_basis, 1, 1, n[d]]

            b = torch.where(mask, x_diff_expanded / denominators_expanded, torch.tensor(1.0, device=self.device))
            b_list.append(torch.prod(b, dim=-1))  # [num_basis, batch, inputs]

        return torch.prod(torch.stack(b_list), dim=0)  # [num_basis, batch, inputs]
    
    def interpolate(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Interpolates the input using the Lagrange basis.
        :param x: size[batch, inputs, dimensions]
        :param w: size[output, inputs, num_basis]
        :returns: size[batch, output]
        """
        basis = self._compute_basis(x, self.indexes)  # [num_basis, batch, inputs]
        out_sum = torch.einsum("ibk,oki->bo", basis, w)  # [batch, output]

        return out_sum

class FourierBasis:
    def __init__(self, length: float):
        """
        Fourier basis functions [sin, cos]
        Args :
            length : the length of the basis function. A value
            of 1 means there is periodicity 1
        """
        self.length = length
        self.num_basis = None  # Apparently defined elsewhere? How does this work!

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


class LagrangeBasis1:
    """
    TODO: Degenerate case, test this and see if it works with everything else.

    """

    def __init__(self, length: float = 2.0):
        self.n = 1
        self.X = torch.tensor([0.0])
        self.num_basis = 1

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

"""
class LagrangePolyFlatND(BasisFlatND):
    def __init__(self, n: int, length: float = 2.0, dimensions: int = 2, **kwargs):
        super().__init__(
            n,
            LagrangeBasisND(n, length, dimensions=dimensions, **kwargs),
            dimensions=dimensions,
            **kwargs
        )
"""

class LagrangePolyFlatND(LagrangeBasisND):
    def __init__(self, n: int, length: float = 2.0, dimensions: int = 2, **kwargs):
        super().__init__(
            n,
            length=length,
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
