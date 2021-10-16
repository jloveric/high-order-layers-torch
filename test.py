import os
import pytest
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.PolynomialLayers import *


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
    assert abs(0.5-ans[0]) < 1.0e-6


@pytest.mark.parametrize("n_in,n_out,in_features,out_features,segments", [(3, 5, 3, 2, 5), (5, 5, 2, 3, 2), (7, 5, 3, 2, 5)])
def test_interpolate_layer(n_in: int, n_out: int, in_features: int, out_features: int, segments: int):
    layer_in = PiecewisePolynomial(
        n=n_in, in_features=in_features, out_features=out_features, segments=segments)
    layer_out = PiecewisePolynomial(
        n=n_out, in_features=in_features, out_features=out_features, segments=segments)
    interpolate_polynomial_layer(layer_in=layer_in, layer_out=layer_out)

    x_in = torch.rand(2, in_features)
    x_out_start = layer_in(x_in)
    x_out_end = layer_out(x_in)

    if n_in <= n_out:  # There should be no loss of information
        assert torch.allclose(x_out_start, x_out_end, rtol=1e-5)
    else:
        pass


def test_compare():

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    stride = 1
    height = 5
    width = 5
    n = 3
    segments = 1

    values = {"n": n, "in_channels": in_channels, "out_channels": out_channels,
              "kernel_size": kernel_size, "stride": stride}

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

    values = {"n": n, "in_channels": in_channels, "out_channels": out_channels,
              "kernel_size": kernel_size, "stride": stride}

    x = torch.rand(1, in_channels, height, width)
    a = PolynomialConvolution2d(**values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2
    assert aout.shape[3] == 2


def test_piecewise_poly_convolution_2d_produces_correct_sizes():

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

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2
    assert aout.shape[3] == 2


def test_discontinuous_poly_convolution_2d_produces_correct_sizes():

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

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2
    assert aout.shape[3] == 2

