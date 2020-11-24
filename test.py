import os

import unittest
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.FunctionalConvolution import *
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
        segments=1

        values = {"n": n, "in_channels": in_channels, "out_channels": out_channels,
                  "kernel_size": kernel_size, "stride": stride}

        x = torch.rand(1, in_channels, height, width)
        a = Expansion2d(LagrangeExpand(n))
        b = Expansion2d(PiecewisePolynomialExpand(n=n, segments=segments))

        aout = a(x)
        bout = b(x)

        assert torch.allclose(aout, bout, atol=1e-5)

class TestConvolution2d(unittest.TestCase):

    def test_poly_convolution_2d_produces_correct_sizes(self):

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
        
        aout = a(x)

        assert aout.shape[0]==1
        assert aout.shape[1]==2
        assert aout.shape[2]==2
        assert aout.shape[3]==2

    def test_piecewise_poly_convolution_2d_produces_correct_sizes(self):

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
        a = PiecewisePolynomialConvolution2d(
            segments=1, **values)

        aout = a(x)
        
        assert aout.shape[0]==1
        assert aout.shape[1]==2
        assert aout.shape[2]==2
        assert aout.shape[3]==2

    def test_discontinuous_poly_convolution_2d_produces_correct_sizes(self):

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
        a = PiecewiseDiscontinuousPolynomialConvolution2d(
            segments=1, **values)

        aout = a(x)
        
        assert aout.shape[0]==1
        assert aout.shape[1]==2
        assert aout.shape[2]==2
        assert aout.shape[3]==2


if __name__ == '__main__':
    unittest.main()
