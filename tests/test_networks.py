import os
import pytest
from high_order_layers_torch.networks import HighOrderFullyConvolutionalNetwork
import torch

@pytest.mark.parametrize(
    "segments,n,kernel_size,ctype",
    [(2, 3,1,"polynomial1d"), (1, 5, 3,"continuous1d")],
)
def test_interpolate_fully_convolutional_network(segments, n, kernel_size,ctype):

    size = 5
    width = 500
    channels = 7
    #kernel_size = 2
    model = HighOrderFullyConvolutionalNetwork(
        layer_type=[ctype] * size,
        n=[n] * size,
        channels=[channels] * (size + 1),
        segments=[segments] * size,
        kernel_size=[kernel_size] * size,
    )

    x = torch.rand(2, channels, width)
    out = model(x)
    print("out", out.shape)

    # y0 = network_in(x)
    # y1 = network_out(x)
    # assert torch.allclose(y0, y1, rtol=1e-4)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == channels
    assert out.shape[2] == width-(kernel_size-1)*size
