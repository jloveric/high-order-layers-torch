import numpy as np
import math
import torch


def chebyshevLobatto(n):

    k = torch.arange(0, n)

    ans = -torch.cos(k * math.pi / (n - 1))

    ans = torch.where(torch.abs(ans) < 1e-15, 0*ans, ans)

    return ans


class LagrangeExpand:

    def __init__(self, n):
        self.n = n
        self.X = chebyshevLobatto(n)

    def basis(self, x, j):

        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        b = torch.stack(b)
        ans = torch.prod(b, dim=0)
        return ans

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
        out_dim = w.shape[2]
        print('x.shape', x.shape)
        print('w.shape', w.shape)
        mat = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            w_j = w[..., j]
            print('basis_j.shape', basis_j.shape, 'w_j.shape', w_j.shape)
            out_list = []

            for b in range(basis_j.shape[0]):
                final = basis_j[b,...]* w_j[..., ].permute(1, 0)
                out_list.append(final)
            print('final.shape', final.shape)
            mat.append(torch.stack(final))

        # Sum up the components to produce the final polynomial
        mat = torch.stack(mat)
        print('mat.shape', mat.shape)
        assemble = torch.sum(torch.stack(mat), dim=0)

        print('assemble.hape', assemble.shape)

        # Compute sum and product at output
        out_sum = torch.sum(assemble, dim=0)
        out_prod = torch.prod(assemble, dim=0)

        ans_sum = torch.transpose(out_sum, 0, 1)
        ans_prod = torch.transpose(out_prod, 0, 1)
        return ans_sum, ans_prod


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
        out_dim = w.shape[2]
        print('x.shape', x.shape)
        print('w.shape', w.shape)
        mat = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            w_j = w[..., j]

            out_list = []
            for out in range(out_dim):
                final = basis_j*w_j[..., out]
                out_list.append(final)

            mat.append(torch.stack(out_list))

        # Sum up the components to produce the final polynomial
        assemble = torch.sum(torch.stack(mat), dim=0)

        # Compute sum and product at output
        out_sum = torch.sum(assemble, dim=2)
        out_prod = torch.prod(assemble, dim=2)

        ans_sum = torch.transpose(out_sum, 0, 1)
        ans_prod = torch.transpose(out_prod, 0, 1)
        return ans_sum, ans_prod
