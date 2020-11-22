import os

import unittest
from functional_layers.LagrangePolynomial import *
from functional_layers.FunctionalConvolution import *
import numpy as np


class TestPolynomials(unittest.TestCase):

    def test_nodes(self):
        ans = chebyshevLobatto(20)
        self.assertTrue(ans.shape[0] == 20)

    def test_polynomial(self):
        poly = LagrangePoly(5)
        # Just use the points as the actual values
        w = chebyshevLobatto(5)
        w = w.reshape(1, 1, 1, 5)
        x = torch.tensor([[0.5]])
        ans = poly.interpolate(x, w)
        self.assertTrue(abs(0.5-ans[0]) < 1.0e-6)


class TestExpansion2d(unittest.TestCase):

    def test_compare(self):

        in_channels = 2
        out_channels = 2
        kernel_size = 4
        stride = 1
        height = 5
        width = 5
        n = 3

        values = {"n": n, "in_channels": in_channels, "out_channels": out_channels,
                  "kernel_size": kernel_size, "stride": stride}

        x = torch.rand(1, in_channels, height, width)
        a = PolynomialConvolution2d(**values)
        b = PiecewisePolynomialConvolution2d(
            segments=1, **values)

        aout = a(x)
        bout = b(x)

        print('aout', aout)
        print('bout', bout)
        print('diff', aout-bout)
        assert torch.all(torch.eq(aout, bout))


if __name__ == '__main__':
    unittest.main()
