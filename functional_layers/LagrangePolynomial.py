import numpy as np
import math
import torch


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

        i = (j+1)/2
        if j % 2 == 0:
            return torch.cos(math.pi*i*x/self.length)
        else:
            return torch.sin(math.pi*i*x/self.length)


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


class BasisExpand :
    def __init__(self, basis, n) :
        self.n = n
        self.basis = basis

    def __call__(self, x) :
        """
        Args:
            - x: size[batch, input]
        Returns:
            - result: size[batch, output]
        """
        mat = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            mat.append(basis_j)

        return torch.stack(mat)

class LagrangeExpand:

    def __init__(self, n):
        self.n = n
        #self.X = chebyshevLobatto(n)
        self.basis = LagrangeBasis(n)

    """
    def basis(self, x, j):

        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        b = torch.stack(b)
        ans = torch.prod(b, dim=0)
        return ans
    """
    
    def __call__(self, x):
        """
        Args:
            - x: size[batch, input]
            - w: size[batch, input, n]
        Returns:
            - result: size[batch, output]
        """
        mat = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            mat.append(basis_j)

        return torch.stack(mat)


class LagrangePolyFlat:
    """
    Single segment.
    """

    def __init__(self, n):
        self.n = n
        self.X = chebyshevLobatto(n)

    def basis(self, x, j):

        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        b = torch.stack(b)
        ans = torch.prod(b, dim=0)
        return ans

    def interpolate(self, x, w):
        """
        Args:
            - x: size[batch, input]
            - w: size[input, output, basis]
        Returns:
            - result: size[batch, output]
        """

        basis = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            basis.append(basis_j)
        basis = torch.stack(basis)
        assemble = torch.einsum("ijk,lki->jlk", basis, w)

        # Compute sum and product at output
        out_sum = torch.sum(assemble, dim=2)
        out_prod = torch.prod(assemble, dim=2)

        return out_sum, out_prod


class LagrangePoly:

    def __init__(self, n):
        self.n = n
        self.X = chebyshevLobatto(n)

    def basis(self, x, j):

        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        b = torch.stack(b)
        ans = torch.prod(b, dim=0)
        return ans

    def interpolate(self, x, w):
        """
        Args:
            - x: size[batch, input]
            - w: size[batch, input, output, basis]
        Returns:
            - result: size[batch, output]
        """

        mat = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            mat.append(basis_j)
        mat = torch.stack(mat)

        assemble = torch.einsum("ijk,jkli->jlk", mat, w)

        # Compute sum and product at output
        out_sum = torch.sum(assemble, dim=2)
        out_prod = torch.prod(assemble, dim=2)

        return out_sum, out_prod
