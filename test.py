import os
 
import unittest
from high_order_layers_torch.LagrangePolynomial import *
import numpy as np

class TestPolynomials(unittest.TestCase):

    def test_nodes(self):
        ans = chebyshevLobatto(20)
        print('ans', ans)

if __name__ == '__main__':
    unittest.main()