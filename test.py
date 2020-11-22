import os
 
import unittest
from functional_layers.LagrangePolynomial import *
import numpy as np


class TestPolynomials(unittest.TestCase):

    def test_nodes(self):
        ans = chebyshevLobatto(20)
        self.assertTrue(ans.shape[0]==20)

    def test_polynomial(self):
        poly = LagrangePoly(5)
        #Just use the points as the actual values
        w = chebyshevLobatto(5)
        w=w.reshape(1,1,1,5)
        x=torch.tensor([[0.5]])
        ans = poly.interpolate(x, w)
        self.assertTrue(abs(0.5-ans[0])<1.0e-6)

class TestExpansion2d(unittest.TestCase) :

    def test_compare(self) :
        pass

if __name__ == '__main__':
    unittest.main()