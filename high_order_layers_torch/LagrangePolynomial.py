import numpy as np
import math

def chebyshevLobatto(int k, int n)
    # rescale range to be between 0 and 1
    if k == 0 :
        return 0.0

    if (k == (n - 1))
        return 1.0

    k = np.arange(0,n)
    

    return 0.5 * (-cos(k * math.pi / (n - 1)) + 1.0)



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