import os

import pytest

from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.layers import (
    L2Normalization,
    MaxAbsNormalization,
    MaxAbsNormalizationND,
    fixed_rotation_layer,
    initialize_polynomial_layer,
)
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


def test_max_abs_layers():

    x = torch.tensor([[1, 0.5, 0.5], [2, 0.5, 0.5]])

    layer = MaxAbsNormalization(eps=0.0)
    ans = layer(x)
    assert torch.all(torch.eq(ans[0], torch.tensor([1, 0.5, 0.5])))
    assert torch.all(torch.eq(ans[1], torch.tensor([1, 0.25, 0.25])))

    layer = MaxAbsNormalizationND(eps=0.0)
    ans = layer(x)
    assert torch.all(torch.eq(ans[0], torch.tensor([1, 0.5, 0.5])))
    assert torch.all(torch.eq(ans[1], torch.tensor([1, 0.25, 0.25])))

    x = torch.tensor([[[1, 0.5, 0.5], [2, 0.5, 0.5]], [[4, 0.5, 0.5], [8, 0.5, 0.5]]])
    ans = layer(x)
    assert torch.all(torch.eq(ans[0][0], torch.tensor([0.5, 0.25, 0.25])))
    assert torch.all(torch.eq(ans[0][1], torch.tensor([1, 0.25, 0.25])))
    assert torch.all(torch.eq(ans[1][0], torch.tensor([0.5, 0.0625, 0.0625])))
    assert torch.all(torch.eq(ans[1][1], torch.tensor([1, 0.0625, 0.0625])))


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("rotations", [1, 2, 3])
def test_fixed_rotation_layer(n: int, rotations: int):
    layer, size = fixed_rotation_layer(n=n, rotations=rotations)

    if n == 2 and rotations == 1:
        assert torch.allclose(
            torch.tensor([[1.0000e00, 0.0000e00], [0.0, 1.0000e00]]),
            layer.weight,
        )
    elif n == 2 and rotations == 2:
        assert torch.allclose(
            torch.tensor(
                [
                    [1.0000, 0.0000],
                    [0.7071, 0.7071],
                    [0.7071, -0.7071],
                    [0.0000, 1.0000],
                ]
            ),
            layer.weight,
        )
    elif (n == 2) and (rotations == 3):
        assert torch.allclose(
            torch.tensor(
                [
                    [1.0000, 0.0000],
                    [0.8660, 0.5000],
                    [0.5000, -0.8660],
                    [0.5000, 0.8660],
                    [0.8660, -0.5000],
                    [0.0000, 1.0000],
                ]
            ),
            layer.weight,
            atol=1.0e-4,
        )
    elif n == 3 and rotations == 1:
        assert torch.allclose(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            layer.weight,
        )
    elif n == 3 and rotations == 2:
        assert torch.allclose(
            torch.tensor(
                [
                    [1.0000, 0.0000, 0.0000],
                    [0.7071, 0.7071, 0.0000],
                    [0.7071, -0.7071, 0.0000],
                    [0.7071, 0.0000, 0.7071],
                    [0.7071, 0.0000, -0.7071],
                    [0.0000, 1.0000, 0.0000],
                    [0.0000, 0.7071, 0.7071],
                    [0.0000, 0.7071, -0.7071],
                    [0.0000, 0.0000, 1.0000],
                ]
            ),
            layer.weight,
            atol=1e-4,
        )

    tsize = n + n * (n - 1) * (rotations - 1)
    assert size == tsize
    assert layer.weight.shape == torch.Size([tsize, n])


@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("in_features", [2, 3])
@pytest.mark.parametrize("out_features", [2, 3])
@pytest.mark.parametrize("segments", [2, 3])
@pytest.mark.parametrize("max_offset", [0.0])
def test_initialize_polynomial(
    n: int, in_features: int, out_features: int, segments: int, max_offset
):
    layer = PiecewisePolynomial(
        n=n, in_features=in_features, out_features=out_features, segments=segments
    )

    initialize_polynomial_layer(layer, max_slope=1.0, max_offset=max_offset)

    assert torch.allclose(layer.w[:, :, 0], -layer.w[:, :, -1])


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("in_features", [2, 3])
@pytest.mark.parametrize("out_features", [2, 3])
@pytest.mark.parametrize("segments", [2, 3])
@pytest.mark.parametrize("max_offset", [0.0])
def test_initialize_discontinuous_polynomial(
    n: int, in_features: int, out_features: int, segments: int, max_offset
):
    layer = PiecewiseDiscontinuousPolynomial(
        n=n, in_features=in_features, out_features=out_features, segments=segments
    )

    initialize_polynomial_layer(layer, max_slope=1.0, max_offset=max_offset)

    assert torch.allclose(layer.w[:, :, 0], -layer.w[:, :, -1])
