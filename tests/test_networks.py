import os

import pytest
import torch

from high_order_layers_torch.networks import (
    HighOrderFullyConvolutionalNetwork,
    HighOrderMLP,
    HighOrderTailFocusNetwork,
    LowOrderMLP,
    initialize_network_polynomial_layers,
    transform_mlp,
)


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("in_width", [1, 3])
@pytest.mark.parametrize("out_width", [1, 3])
@pytest.mark.parametrize(
    "ctype",
    [
        "polynomial",
        "continuous",
        "discontinuous",
        "switch_continuous",
        "switch_discontinuous",
    ],
)
@pytest.mark.parametrize("hidden_width", [1, 3])
@pytest.mark.parametrize("hidden_layers", [1, 3])
def test_fully_connected_network(
    segments, n, in_width, out_width, ctype, hidden_width, hidden_layers
):
    width = 100

    model = HighOrderMLP(
        in_segments=segments,
        out_segments=segments,
        hidden_segments=segments,
        layer_type=ctype,
        n=n,
        in_width=in_width,
        out_width=out_width,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        normalization=None,
    )

    x = torch.rand(2, in_width)
    out = model(x)

    assert out.shape == torch.Size([2, out_width])
    assert True is True


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
@pytest.mark.parametrize("resnet", [True, False])
@pytest.mark.parametrize("rotations", [1, 3])
def test_transform_mlp(layer_type: str, resnet: bool, rotations: int):
    in_width = 3
    out_width = 2
    n = 3
    network = transform_mlp(
        layer_type=layer_type,
        n=3,
        n_in=n,
        n_hidden=n,
        n_out=n,
        in_width=in_width,
        out_width=out_width,
        hidden_layers=2,
        hidden_width=4,
        hidden_segments=2,
        in_segments=2,
        out_segments=2,
        resnet=resnet,
        rotations=rotations,
    )

    x = torch.rand(2, in_width)
    s = network(x)

    # Not much of a test, just verify nothing crashes the code
    assert isinstance(s, torch.Tensor)


@pytest.mark.parametrize(
    "layer_type",
    ["continuous", "discontinuous", "switch_continuous", "switch_discontinuous"],
)
@pytest.mark.parametrize("resnet", [True, False])
def test_initialize_network_polynomial_layers(layer_type: str, resnet: bool):
    in_width = 3
    out_width = 2
    network = HighOrderMLP(
        layer_type=layer_type,
        n=3,
        in_width=in_width,
        out_width=out_width,
        hidden_layers=2,
        hidden_width=4,
        hidden_segments=2,
        in_segments=2,
        out_segments=2,
        resnet=resnet,
    )

    x = torch.rand(2, in_width)
    s = network(x)
    initialize_network_polynomial_layers(network=network, max_slope=1.0, max_offset=0.0)
    y = network(x)

    assert not torch.allclose(s, y)


def test_ignores_non_polynomial_layers():
    in_width = 3
    out_width = 2
    network = LowOrderMLP(
        in_width=in_width,
        out_width=out_width,
        hidden_layers=2,
        hidden_width=4,
    )

    x = torch.rand(2, in_width)
    s = network(x)
    initialize_network_polynomial_layers(network=network, max_slope=1.0, max_offset=0.0)
    y = network(x)

    assert torch.allclose(s, y)
