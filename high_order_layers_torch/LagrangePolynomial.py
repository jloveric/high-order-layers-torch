# copied and modified from https://stackoverflow.com/questions/4003794/lagrange-interpolation-in-python
# Should simplify this for Gauss Lobatto points explicitly
import numpy as np

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