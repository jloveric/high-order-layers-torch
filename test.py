import os
 
import unittest
from high_order_layers_torch.LagrangePolynomial import *
import numpy as np

class TestPolynomials(unittest.TestCase):

    def test_nodes(self):
        ans = chebyshevLobatto(20)
        print('ans', ans)
        self.assertTrue(ans.shape[0]==20)

    def test_polynomial(self):
        poly = LagrangePoly(5)
        #Just use the points as the actual values
        w = chebyshevLobatto(5)
        print('w', w)
        ans = poly.interpolate(0.5, w)
        print('ans', ans)
        self.assertTrue(abs(0.5-ans)<1.0e-6)

if __name__ == '__main__':
    unittest.main()