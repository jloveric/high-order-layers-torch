import os
import pytest
from high_order_layers_torch.networks import HighOrderFullyConvolutionalNetwork
import torch


@pytest.mark.parametrize(
    "segments,n",
    [(2, 3), (2, 5)],
)
def test_interpolate_fully_convolutional_network(segments, n):

    size = 5
    channels = 7
    kernel_size = 1
    model = HighOrderFullyConvolutionalNetwork(
        layer_type=["polynomial1d"] * size,
        n=[n] * size,
        channels=[channels] * (size + 1),
        segments=[segments] * size,
        kernel_size=[kernel_size] * size,
    )

    x = torch.rand(2, 5, 7)
    out = model(x)
    print("out", out)

    # y0 = network_in(x)
    # y1 = network_out(x)
    # assert torch.allclose(y0, y1, rtol=1e-4)
    assert True is False
