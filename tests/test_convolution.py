import os

import pytest

import torch
from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.LagrangePolynomial import *

torch.set_default_device(device="cpu")


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
        "device": "cpu",
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
        "device": "cpu",
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
        "device": "cpu",
    }

    x = torch.rand(1, in_channels, height, width)
    a = PiecewisePolynomialConvolution2d(segments=4, **values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2
    assert aout.shape[3] == 2


# Currently broken!
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
        "device": "cpu",
    }

    x = torch.rand(1, in_channels, height)
    a = PiecewisePolynomialConvolution1d(segments=1, **values)

    aout = a(x)

    print("a.out.shape", aout.shape)

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 2


def test_discontinuous_poly_convolution_2d_produces_correct_sizes():
    in_channels = 2
    out_channels = 2
    kernel_size = 4
    stride = 1
    height = 6
    width = 7
    n = 3

    values = {
        "n": n,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "device": "cpu",
    }

    x = torch.rand(1, in_channels, height, width)
    a = PiecewiseDiscontinuousPolynomialConvolution2d(segments=1, **values)

    aout = a(x)

    assert aout.shape[0] == 1
    assert aout.shape[1] == 2
    assert aout.shape[2] == 3
    assert aout.shape[3] == 4
