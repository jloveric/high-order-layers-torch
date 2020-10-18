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
        # TODO: this can probably be made more efficient
        # doing it this way since the multiplication keeps
        # broadcasting where I don't want it to.
        print('x.shape', x.shape)
        batch_list = []
        for batch in range(x.shape[0]):
            out_list = []
            for out in range(w.shape[2]):
                b = 0
                for inp in range(x.shape[1]):
                    b += sum([self.basis(x[batch, inp], j)*w[batch, inp, out, j]
                             for j in range(self.n)])
                out_list.append(b)

            batch_list.append(torch.tensor(out_list,requires_grad=True))

        ans = torch.stack(batch_list)
        print('ans.shape', ans.shape)
        return ans
