import numpy as np
import math

def chebyshevLobatto(n) :

    k = np.arange(0,n)

    ans =  (-np.cos(k * math.pi / (n - 1)) + 1.0)-1.0

    ans = np.where(np.abs(ans)<1e-15,0,ans)

    return ans



class LagrangePoly:

    def __init__(self, X, Y):
        self.n = len(X)
        self.X = np.array(X)

    def basis(self, x, j):
        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        return np.prod(b, axis=0)

    def interpolate(self, x, w):
        b = [self.basis(x, j)*w[j] for j in range(self.n)]
        return np.sum(b, axis=0)