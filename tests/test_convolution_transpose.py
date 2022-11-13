import os

import pytest

from high_order_layers_torch.FunctionalConvolutionTranspose import *


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("height", [1, 3])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("stride", [1, 2])
def test_piecewise_poly_convolution_transpose_2d_produces_correct_sizes(
    segments, n, kernel_size, in_channels, out_channels, height, width, stride
):

    output_padding = 0
    padding = 0

    values = {
        "n": n,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "output_padding": output_padding,
        "padding": padding,
        "segment": segments,
    }

    x = torch.rand(1, in_channels, height, width)
    a = PiecewisePolynomialConvolutionTranspose2d(segments=4, **values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == out_channels
    assert aout.shape[2] == stride * (height - 1) + kernel_size - 2 * padding
    assert aout.shape[3] == stride * (width - 1) + kernel_size - 2 * padding
