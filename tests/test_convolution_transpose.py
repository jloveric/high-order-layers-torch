import os
import pytest
from high_order_layers_torch.FunctionalConvolutionTranspose import *


def test_piecewise_poly_convolution_transpose_2d_produces_correct_sizes():

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    output_padding = 0
    padding = 0
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
        "output_padding": output_padding,
        "padding": padding,
    }

    x = torch.rand(1, in_channels, height, width)
    a = PiecewisePolynomialConvolutionTranspose2d(segments=4, **values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == out_channels
    assert aout.shape[2] == height + kernel_size - 1
    assert aout.shape[3] == width + kernel_size - 1
