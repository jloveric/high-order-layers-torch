import os

import pytest
import torch

from high_order_layers_torch.networks import (
    HighOrderFullyConvolutionalNetwork,
    HighOrderTailFocusNetwork,
    HighOrderMLP,
    transform_mlp,
    initialize_network_polynomial_layers,
)


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("ctype", ["polynomial1d", "continuous1d", "discontinuous1d"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("layers", [1, 3])
def test_fully_convolutional_network1d(
    segments, n, kernel_size, ctype, channels, layers
):
    width = 100

    model = HighOrderFullyConvolutionalNetwork(
        layer_type=[ctype] * layers,
        n=[n] * layers,
        channels=[channels] * (layers + 1),
        segments=[segments] * layers,
        kernel_size=[kernel_size] * layers,
        pooling="1d",
    )

    x = torch.rand(2, channels, width)
    out = model(x)
    print("out", out.shape)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == channels


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("ctype", ["polynomial2d", "continuous2d", "discontinuous2d"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("layers", [1, 3])
def test_fully_convolutional_network2d(
    segments, n, kernel_size, ctype, channels, layers
):
    width = 100

    model = HighOrderFullyConvolutionalNetwork(
        layer_type=[ctype] * layers,
        n=[n] * layers,
        channels=[channels] * (layers + 1),
        segments=[segments] * layers,
        kernel_size=[kernel_size] * layers,
        pooling="2d",
    )

    x = torch.rand(2, channels, width, width)
    out = model(x)
    print("out", out.shape)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == channels


@pytest.mark.parametrize("segments", [2])
@pytest.mark.parametrize("n", [3])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("ctype", ["polynomial2d"])
@pytest.mark.parametrize("channels", [3])
@pytest.mark.parametrize("layers", [3])
def test_convolutional_network_no_pool2d(
    segments, n, kernel_size, ctype, channels, layers
):
    width = 100

    model = HighOrderFullyConvolutionalNetwork(
        layer_type=[ctype] * layers,
        n=[n] * layers,
        channels=[channels] * (layers + 1),
        segments=[segments] * layers,
        kernel_size=[kernel_size] * layers,
        pooling=None,
    )

    x = torch.rand(2, channels, width, width)
    out = model(x)
    print("out", out.shape)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == 26508


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("ctype", ["polynomial1d", "continuous1d", "discontinuous1d"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("layers", [1, 3])
def test_convolutional_network_with_stride_list(
    segments, n, kernel_size, ctype, channels, layers
):
    width = 100

    model = HighOrderFullyConvolutionalNetwork(
        layer_type=[ctype] * layers,
        n=[n] * layers,
        channels=[channels] * (layers + 1),
        segments=[segments] * layers,
        kernel_size=[kernel_size] * layers,
        pooling="1d",
        stride=[2] * layers,
    )

    x = torch.rand(2, channels, width)
    out = model(x)
    print("out", out.shape)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == channels


@pytest.mark.parametrize("segments", [2])
@pytest.mark.parametrize("n", [3])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("ctype", ["continuous1d"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("layers", [1, 2, 3])
def test_tail_focus(segments, n, kernel_size, ctype, channels, layers):
    width = 100

    model = HighOrderTailFocusNetwork(
        layer_type=[ctype] * layers,
        n=[n] * layers,
        channels=[channels] * (layers + 1),
        segments=[segments] * layers,
        kernel_size=[kernel_size] * layers,
        stride=[2] * layers,
        focus=[2] * layers,
        normalization=torch.nn.LazyBatchNorm1d,
    )
    width_list, output_size = model.compute_sizes(width)
    x = torch.rand(2, channels, width)
    out = model(x)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == sum(output_size)


@pytest.mark.parametrize("layer_type", ["continuous", "discontinuous"])
def test_initialize_network_polynomial_layers(layer_type: str):

    in_width=3
    out_width=2
    network = HighOrderMLP(
        layer_type=layer_type,
        n=3,
        in_width=in_width,
        out_width=out_width,
        hidden_layers=2,
        hidden_width=4,
        hidden_segments=2,
        in_segments=2,
        out_segments=2
    )
    initialize_network_polynomial_layers(network=network, max_slope=1.0, max_offset=0.0)
    x = torch.rand(2, in_width)
    y = network(x)

    # Ok, this tests nothing other than the thing runs
    assert torch.Size([2,2])==y.shape
