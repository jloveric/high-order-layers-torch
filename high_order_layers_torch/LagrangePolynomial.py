import numpy as np
import math
import torch


def chebyshevLobatto(n):

    k = torch.arange(0, n)

    ans = -torch.cos(k * math.pi / (n - 1))

    ans = torch.where(torch.abs(ans) < 1e-15, 0*ans, ans)

    return ans


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
        # Todo this can probably be made more efficient
        b = [self.basis(x, j)*w[:, :,j] for j in range(self.n)]
        b = torch.stack(b)
        return torch.sum(b, dim=0)