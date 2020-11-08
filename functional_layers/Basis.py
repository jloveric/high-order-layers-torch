import numpy as np
import math
import torch


class BasisExpand:
    def __init__(self, basis, n):
        self.n = n
        self.basis = basis

    def __call__(self, x):
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


class BasisFlat:
    """
    Single segment.
    """

    def __init__(self, n, basis):
        self.n = n
        self.basis = basis

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

# Is this the same as above?
class Basis:

    def __init__(self, n, basis):
        self.n = n
        self.basis = basis

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