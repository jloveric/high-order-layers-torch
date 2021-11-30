import os
import pytest
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.PolynomialLayers import *
from high_order_layers_torch.networks import *


def test_nodes():
    ans = chebyshevLobatto(20)
    assert ans.shape[0] == 20


def test_polynomial():
    poly = LagrangePoly(5)
    # Just use the points as the actual values
    w = chebyshevLobatto(5)
    w = w.reshape(1, 1, 1, 5)
    x = torch.tensor([[0.5]])
    ans = poly.interpolate(x, w)
    assert abs(0.5 - ans[0]) < 1.0e-6


def test_compare():

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    stride = 1
    height = 5
    width = 5
    n = 3
    segments = 1

    values = {
        "n": n,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
    }

    x = torch.rand(1, in_channels, height, width)
    a = Expansion2d(LagrangeExpand(n))
    b = Expansion2d(PiecewisePolynomialExpand(n=n, segments=segments))

    aout = a(x)
    bout = b(x)

    assert torch.allclose(aout, bout, atol=1e-5)


def test_poly_convolution_2d_produces_correct_sizes():

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    stride = 1
    height = 5
    width = 5
    n = 3

    values = {
        "n": n,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
    }

    x = torch.rand(1, in_channels, height, width)
    a = PolynomialConvolution2d(**values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2
    assert aout.shape[3] == 2


def test_poly_convolution_1d_produces_correct_sizes():

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    stride = 1
    width = 5
    n = 3

    values = {
        "n": n,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
    }

    x = torch.rand(1, in_channels, width)
    a = PolynomialConvolution1d(**values)

    aout = a(x)
    print("aout.shape", aout.shape)
    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2


def test_piecewise_poly_convolution_2d_produces_correct_sizes():

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    stride = 1
    height = 5
    width = 5
    n = 3

    values = {
        "n": n,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
    }

    x = torch.rand(1, in_channels, height, width)
    a = PiecewisePolynomialConvolution2d(segments=1, **values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2
    assert aout.shape[3] == 2


"""
Currently broken!
def test_piecewise_poly_convolution_1d_produces_correct_sizes():

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    stride = 1
    height = 5
    width = 5
    n = 3

    values = {
        "n": n,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
    }

    x = torch.rand(1, in_channels, height)
    a = PiecewisePolynomialConvolution1d(segments=1, **values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2
    assert aout.shape[3] == 2
"""


def test_discontinuous_poly_convolution_2d_produces_correct_sizes():

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    stride = 1
    height = 5
    width = 5
    n = 3

    values = {
        "n": n,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
    }

    x = torch.rand(1, in_channels, height, width)
    a = PiecewiseDiscontinuousPolynomialConvolution2d(segments=1, **values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2
    assert aout.shape[3] == 2
