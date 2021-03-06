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

        # TODO: Try and do this as a vector operation.
        mat = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            mat.append(basis_j)

        return torch.stack(mat)


class PiecewiseExpand:
    def __init__(self, basis, n: int, segments: int, length: float = 2.0, sparse=False):
        super().__init__()
        self._basis = basis
        self._n = n
        self._segments = segments
        self._expand = BasisExpand(basis, n)
        self._variables = (self._n-1)*self._segments+1
        self._length = length
        self._half = 0.5*length

    def __call__(self, x):
        """
        Apply basis function to each input.
        Args :
            x : Tensor of shape [batch, channels, x, y]
        Out :
            Tensor of shape [variables, batch, channels, x, y]
        """
        # get the segment index
        id_min = (((x+self._half)/self._length)*self._segments).long()
        device = x.device
        id_min = torch.where(id_min <= self._segments-1, id_min,
                             torch.tensor(self._segments-1, device=device))
        id_min = torch.where(id_min >= 0, id_min,
                             torch.tensor(0, device=device))
        id_max = id_min+1

        wid_min = id_min*(self._n-1)

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length*((x-x_min)/(x_max-x_min))-self._half

        # These are the outputs, but they need to be in a sparse tensor
        # so they work with everything, do dense for now.
        out = self._expand(x_in)

        mat = torch.zeros(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self._variables,
            device=device
        )

        wid_min_flat = wid_min.view(-1)

        wrange = wid_min_flat.unsqueeze(-1) + \
            torch.arange(self._n, device=device).view(-1)

        # This needs to be
        windex = (torch.arange(
            wrange.numel())//self._n)

        out = out.permute(1, 2, 3, 4, 0)

        mat_trans = mat.reshape(-1, self._variables)
        mat_trans[windex, wrange.view(-1)] = out.flatten()
        mat = mat_trans.reshape(
            mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3], mat.shape[4])

        mat = mat.permute(4, 0, 1, 2, 3)

        return mat

    def _eta(self, index):
        """
        Arg:
            - index is the segment index
        """
        eta = index/float(self._segments)
        return eta*2-1


class PiecewiseDiscontinuousExpand:
    # TODO: This and the PiecewiseExpand should share more data.
    def __init__(self, basis, n, segments, length: int = 2.0):
        super().__init__()
        self._basis = basis
        self._n = n
        self._segments = segments
        self._expand = BasisExpand(basis, n)
        self._variables = self._n*self._segments
        self._length = length
        self._half = 0.5*length

    def __call__(self, x):
        """
        Apply basis function to each input.
        Args :
            x : Tensor of shape [batch, channels, x, y]
        Out :
            Tensor of shape [variables, batch, channels, x, y]
        """
        # get the segment index
        id_min = (((x+self._half)/self._length)*self._segments).long()
        device = x.device
        id_min = torch.where(id_min <= self._segments-1, id_min,
                             torch.tensor(self._segments-1, device=device))
        id_min = torch.where(id_min >= 0, id_min,
                             torch.tensor(0, device=device))
        id_max = id_min+1

        wid_min = id_min*self._n

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length*((x-x_min)/(x_max-x_min))-self._half

        # These are the outputs, but they need to be in a sparse tensor
        # so they work with everything, do dense for now.
        out = self._expand(x_in)

        mat = torch.zeros(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self._variables,
            device=device
        )

        wid_min_flat = wid_min.view(-1)

        wrange = wid_min_flat.unsqueeze(-1) + \
            torch.arange(self._n, device=device).view(-1)

        # This needs to be
        windex = (torch.arange(
            wrange.numel())//self._n)

        out = out.permute(1, 2, 3, 4, 0)

        mat_trans = mat.reshape(-1, self._variables)
        mat_trans[windex, wrange.view(-1)] = out.flatten()
        mat = mat_trans.reshape(
            mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3], mat.shape[4])

        mat = mat.permute(4, 0, 1, 2, 3)

        return mat

    def _eta(self, index):
        """
        Arg:
            - index is the segment index
        """
        eta = index/float(self._segments)
        return eta*2-1


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

        return out_sum


class BasisFlatProd:
    """
    Single segment.
    """

    def __init__(self, n, basis, alpha=1.0):
        self.n = n
        self.basis = basis
        self.alpha = alpha

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
        this_sum = torch.sum(assemble, dim=2)

        assemble = assemble+1

        # Compute sum and product at output
        out_prod = torch.prod(assemble, dim=2)-1+(1-self.alpha) * this_sum

        return out_prod


class Basis:
    # TODO: Is this the same as above? No! It is not!
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

        return out_sum


class BasisProd:
    # TODO: Is this the same as above?
    def __init__(self, n, basis, alpha=1.0):
        self.n = n
        self.basis = basis
        self.alpha = alpha

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

        # jlk is [batch, output, input]
        # batch=j, output=l, input=k, i=basis
        assemble = torch.einsum("ijk,jkli->jlk", mat, w)
        this_sum = torch.sum(assemble, dim=2)
        assemble = 1+assemble

        # Compute the product possibly removing the linear term.
        out_prod = torch.prod(assemble, dim=2)-1+(1-self.alpha) * this_sum

        return out_prod
