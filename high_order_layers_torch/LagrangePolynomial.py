import numpy as np
import math
import torch

def chebyshevLobatto(n) :

    k = torch.arange(0,n)

    ans =  (-torch.cos(k * math.pi / (n - 1)) + 1.0)-1.0

    ans = torch.where(torch.abs(ans)<1e-15,0*ans,ans)

    return ans


class LagrangePoly:

    def __init__(self, n):
        self.n = n
        self.X = chebyshevLobatto(n)
        #print('self.X', self.X)

    def basis(self, x, j):

        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        #print('b',b)
        b=torch.stack(b)
        #print('b', b)
        ans =  torch.prod(b, dim=0)
        #print('ans', ans)
        return ans

    def interpolate(self, x, w):
        #print('x.size()',x.size(),'w.size',w.size(),'n',self.n)
        #print('x',x,'w',w)
        b = [self.basis(x, j)*w[:,j] for j in range(self.n)]
        b = torch.stack(b)
        return torch.sum(b, dim=0)