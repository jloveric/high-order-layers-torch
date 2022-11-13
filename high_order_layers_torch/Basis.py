import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor


class BasisExpand:
    def __init__(self, basis: Callable[[Tensor, int], float], n: int):
        """
        Compute the values of the basis functions given a value x
        Args :
            basis : The basis function (1,x,x*x,...)
            n : Number of elements in the basis set
        """
        self.n = n
        self.basis = basis

    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            - x: size[batch, input]
        Returns:Piecewise
            - result: size[batch, output]
        """

        # TODO: Try and do this as a vector operation.
        # should be able to do this with vmap
        mat = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            mat.append(basis_j)

        return torch.stack(mat)


class PiecewiseExpand:
    def __init__(
        self,
        basis: Callable[[Tensor, int], float],
        n: int,
        segments: int,
        length: float = 2.0,
        sparse=False,
    ):
        """
        Expand a piecewise polynomial into basis values.  Only
        one of the pieces will need to be expanded for each of
        the inputs as only one will contain the value of interest.
        """
        super().__init__()
        self._basis = basis
        self._n = n
        self._segments = segments
        self._expand = BasisExpand(basis, n)
        self._variables = (self._n - 1) * self._segments + 1
        self._length = length
        self._half = 0.5 * length

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply basis function to each input.
        Args :
            x : Tensor of shape [batch, channels, x, y]
        Out :
            Tensor of shape [variables, batch, channels, x, y]
        """
        # get the segment index
        id_min = (((x + self._half) / self._length) * self._segments).long()
        device = x.device
        id_min = torch.where(
            id_min <= self._segments - 1,
            id_min,
            torch.tensor(self._segments - 1, device=device),
        )
        id_min = torch.where(id_min >= 0, id_min, torch.tensor(0, device=device))
        id_max = id_min + 1

        wid_min = id_min * (self._n - 1)

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length * ((x - x_min) / (x_max - x_min)) - self._half

        # These are the outputs, but they need to be in a sparse tensor
        # so they work with everything, do dense for now.
        out = self._expand(x_in)

        mat = torch.zeros(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self._variables,
            device=device,
        )

        wid_min_flat = wid_min.reshape(-1)

        wrange = wid_min_flat.unsqueeze(-1) + torch.arange(self._n, device=device).view(
            -1
        )

        # This needs to be
        windex = torch.div(torch.arange(wrange.numel()), self._n, rounding_mode="floor")

        out = out.permute(1, 2, 3, 4, 0)

        mat_trans = mat.reshape(-1, self._variables)
        mat_trans[windex, wrange.view(-1)] = out.flatten()
        mat = mat_trans.reshape(
            mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3], mat.shape[4]
        )

        mat = mat.permute(4, 0, 1, 2, 3)

        return mat

    def _eta(self, index: Tensor) -> Tensor:
        """
        Arg:
            - index is the segment index
        """
        eta = index / float(self._segments)
        return eta * 2 - 1


class PiecewiseExpand1d:
    def __init__(
        self,
        basis: Callable[[Tensor, int], float],
        n: int,
        segments: int,
        length: float = 2.0,
        sparse: bool = False,
    ):
        super().__init__()
        self._basis = basis
        self._n = n
        self._segments = segments
        self._expand = BasisExpand(basis, n)
        self._variables = (self._n - 1) * self._segments + 1
        self._length = length
        self._half = 0.5 * length

    def __call__(self, x):
        """
        Apply basis function to each input.
        Args :
            x : Tensor of shape [batch, channels, x]
        Out :
            Tensor of shape [variables, batch, channels, x]
        """
        # get the segment index
        id_min = (((x + self._half) / self._length) * self._segments).long()
        device = x.device
        id_min = torch.where(
            id_min <= self._segments - 1,
            id_min,
            torch.tensor(self._segments - 1, device=device),
        )
        id_min = torch.where(id_min >= 0, id_min, torch.tensor(0, device=device))
        id_max = id_min + 1

        wid_min = id_min * (self._n - 1)

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length * ((x - x_min) / (x_max - x_min)) - self._half

        # These are the outputs, but they need to be in a sparse tensor
        # so they work with everything, do dense for now.
        out = self._expand(x_in)

        mat = torch.zeros(
            x.shape[0], x.shape[1], x.shape[2], self._variables, device=device
        )

        wid_min_flat = wid_min.view(-1)

        wrange = wid_min_flat.unsqueeze(-1) + torch.arange(self._n, device=device).view(
            -1
        )

        # This needs to be
        windex = torch.div(torch.arange(wrange.numel()), self._n, rounding_mode="floor")

        out = out.permute(1, 2, 3, 0)

        mat_trans = mat.reshape(-1, self._variables)
        mat_trans[windex, wrange.view(-1)] = out.flatten()
        mat = mat_trans.reshape(mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3])

        mat = mat.permute(3, 0, 1, 2)

        return mat

    def _eta(self, index: Tensor) -> Tensor:
        """
        Arg:
            - index is the segment index
        """
        eta = index / float(self._segments)
        return eta * 2 - 1


class PiecewiseDiscontinuousExpand:
    # TODO: This and the PiecewiseExpand should share more data.
    def __init__(
        self,
        basis: Callable[[Tensor, int], float],
        n: int,
        segments: int,
        length: int = 2.0,
    ):
        super().__init__()
        self._basis = basis
        self._n = n
        self._segments = segments
        self._expand = BasisExpand(basis, n)
        self._variables = self._n * self._segments
        self._length = length
        self._half = 0.5 * length

    def __call__(self, x):
        """
        Apply basis function to each input.
        Args :
            x : Tensor of shape [batch, channels, x, y]
        Out :
            Tensor of shape [variables, batch, channels, x, y]
        """
        # get the segment index
        id_min = (((x + self._half) / self._length) * self._segments).long()
        device = x.device
        id_min = torch.where(
            id_min <= self._segments - 1,
            id_min,
            torch.tensor(self._segments - 1, device=device),
        )
        id_min = torch.where(id_min >= 0, id_min, torch.tensor(0, device=device))
        id_max = id_min + 1

        wid_min = id_min * self._n

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length * ((x - x_min) / (x_max - x_min)) - self._half

        # These are the outputs, but they need to be in a sparse tensor
        # so they work with everything, do dense for now.
        out = self._expand(x_in)

        mat = torch.zeros(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self._variables,
            device=device,
        )

        wid_min_flat = wid_min.reshape(-1)

        wrange = wid_min_flat.unsqueeze(-1) + torch.arange(self._n, device=device).view(
            -1
        )

        # This needs to be
        windex = torch.div(torch.arange(wrange.numel()), self._n, rounding_mode="floor")

        out = out.permute(1, 2, 3, 4, 0)

        mat_trans = mat.reshape(-1, self._variables)
        mat_trans[windex, wrange.view(-1)] = out.flatten()
        mat = mat_trans.reshape(
            mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3], mat.shape[4]
        )

        mat = mat.permute(4, 0, 1, 2, 3)

        return mat

    def _eta(self, index: Tensor) -> Tensor:
        """
        Arg:
            - index is the segment index
        """
        eta = index / float(self._segments)
        return eta * 2 - 1


class PiecewiseDiscontinuousExpand1d:
    # TODO: This and the PiecewiseExpand should share more data.
    def __init__(
        self,
        basis: Callable[[Tensor, int], float],
        n: int,
        segments: int,
        length: int = 2.0,
    ):
        super().__init__()
        self._basis = basis
        self._n = n
        self._segments = segments
        self._expand = BasisExpand(basis, n)
        self._variables = self._n * self._segments
        self._length = length
        self._half = 0.5 * length

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply basis function to each input.
        Args :
            x : Tensor of shape [batch, channels, x, y]
        Out :
            Tensor of shape [variables, batch, channels, x, y]
        """
        # get the segment index
        id_min = (((x + self._half) / self._length) * self._segments).long()
        device = x.device
        id_min = torch.where(
            id_min <= self._segments - 1,
            id_min,
            torch.tensor(self._segments - 1, device=device),
        )
        id_min = torch.where(id_min >= 0, id_min, torch.tensor(0, device=device))
        id_max = id_min + 1

        wid_min = id_min * self._n

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length * ((x - x_min) / (x_max - x_min)) - self._half

        # These are the outputs, but they need to be in a sparse tensor
        # so they work with everything, do dense for now.
        out = self._expand(x_in)

        mat = torch.zeros(
            x.shape[0], x.shape[1], x.shape[2], self._variables, device=device
        )

        wid_min_flat = wid_min.view(-1)

        wrange = wid_min_flat.unsqueeze(-1) + torch.arange(self._n, device=device).view(
            -1
        )

        # This needs to be
        windex = torch.div(torch.arange(wrange.numel()), self._n, rounding_mode="floor")

        out = out.permute(1, 2, 3, 0)

        mat_trans = mat.reshape(-1, self._variables)
        mat_trans[windex, wrange.view(-1)] = out.flatten()
        mat = mat_trans.reshape(mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3])

        mat = mat.permute(3, 0, 1, 2)

        return mat

    def _eta(self, index: int):
        """
        Arg:
            - index is the segment index
        """
        eta = index / float(self._segments)
        return eta * 2 - 1


class BasisFlat:
    """
    Single segment.
    """

    def __init__(self, n: int, basis: Callable[[Tensor, int], float]):
        self.n = n
        self.basis = basis

    def interpolate(self, x: Tensor, w: Tensor) -> Tensor:
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

        return out_sum


class BasisFlatProd:
    """
    Single segment.
    """

    def __init__(
        self, n: int, basis: Callable[[Tensor, int], float], alpha: float = 1.0
    ):
        self.n = n
        self.basis = basis
        self.alpha = alpha

    def interpolate(self, x: Tensor, w: Tensor) -> Tensor:
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
        this_sum = torch.sum(assemble, dim=2)

        assemble = assemble + 1

        # Compute sum and product at output
        out_prod = torch.prod(assemble, dim=2) - 1 + (1 - self.alpha) * this_sum

        return out_prod


class Basis:
    # TODO: Is this the same as above? No! It is not!
    def __init__(self, n: int, basis: Callable[[Tensor, int], float]):
        self.n = n
        self.basis = basis

    def interpolate(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Interpolate based on batched weights which is necessary for piecewise
        networks.
        Args:
            - x: size[batch, input]
            - w: size[batch, input, output, basis] weights are "batched" because
                the piecewise nature means that each input activates a different set
                of weights.
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

        return out_sum


class BasisProd:
    # TODO: Is this the same as above?
    def __init__(
        self, n: int, basis: Callable[[Tensor, int], float], alpha: float = 1.0
    ):
        self.n = n
        self.basis = basis
        self.alpha = alpha

    def interpolate(self, x: Tensor, w: Tensor) -> Tensor:
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

        # jlk is [batch, output, input]
        # batch=j, output=l, input=k, i=basis
        assemble = torch.einsum("ijk,jkli->jlk", mat, w)
        this_sum = torch.sum(assemble, dim=2)
        assemble = 1 + assemble

        # Compute the product possibly removing the linear term.
        out_prod = torch.prod(assemble, dim=2) - 1 + (1 - self.alpha) * this_sum

        return out_prod
