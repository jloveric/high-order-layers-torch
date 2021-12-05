import os
import pytest
from high_order_layers_torch.networks import HighOrderFullyConvolutionalNetwork
import torch


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("ctype", ["polynomial1d", "continuous1d", "discontinuous1d"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("layers", [1, 3])
def test_interpolate_fully_convolutional_network(
    segments, n, kernel_size, ctype, channels, layers
):

    #size = 5
    width = 100

    model = HighOrderFullyConvolutionalNetwork(
        layer_type=[ctype] * layers,
        n=[n] * layers,
        channels=[channels] * (layers + 1),
        segments=[segments] * layers,
        kernel_size=[kernel_size] * layers,
    )

    x = torch.rand(2, channels, width)
    out = model(x)
    print("out", out.shape)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == channels
    assert out.shape[2] == width - (kernel_size - 1) * layers
