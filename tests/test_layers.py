import os

import pytest

from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.networks import *
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


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("in_features", [1, 2, 3])
@pytest.mark.parametrize("out_features", [1, 2, 3])
@pytest.mark.parametrize("segments", [2, 3, 4])
def test_smooth_discontinuous_layer(n, in_features, out_features, segments):
    layer = PiecewiseDiscontinuousPolynomial(
        n=n, in_features=in_features, out_features=out_features, segments=segments
    )

    # A factor of 1 will make the edges continuous.
    smooth_discontinuous_layer(layer=layer, factor=1.0)

    left = layer.w[:, :, (n - 1) : -1 : n]
    right = layer.w[:, :, n:-1:n]

    assert torch.all(torch.isclose(left, right, rtol=1e-3))
