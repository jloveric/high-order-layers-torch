import os

import pytest
from torch.nn import LazyBatchNorm1d, LazyInstanceNorm1d

from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.networks import *
from high_order_layers_torch.PolynomialLayers import *


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("in_width", [5])
@pytest.mark.parametrize("out_width", [1, 4])
@pytest.mark.parametrize("hidden_layers", [1, 3])
@pytest.mark.parametrize("hidden_width", [1, 5])
@pytest.mark.parametrize("n0", [2, 3])
@pytest.mark.parametrize("normalization", [LazyBatchNorm1d])
@pytest.mark.parametrize("layer_type", ["continuous", "baseline", "baseline_relu"])
@pytest.mark.parametrize("nonlinearity", [None, torch.nn.ReLU])
def test_high_order_mlp(
    segments,
    in_width,
    out_width,
    hidden_layers,
    hidden_width,
    n0,
    normalization,
    layer_type,
    nonlinearity,
):
    # You don't need to add a nonlinearity unless you are using "baseline" which is the standard
    # torch.nn.Linear or if you only have a single segment

    network = HighOrderMLP(
        layer_type=layer_type,
        n=n0,
        in_width=in_width,
        out_width=out_width,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        n_in=n0,
        n_out=n0,
        n_hidden=n0,
        in_segments=segments,
        out_segments=segments,
        hidden_segments=segments,
        non_linearity=None if nonlinearity is None else nonlinearity(),
        normalization=normalization,
    )

    batch_size = 4
    x = torch.rand(batch_size, in_width)
    y0 = network(x)
    assert y0.shape[0] == batch_size
    assert y0.shape[1] == out_width


@pytest.mark.parametrize("in_width", [5])
@pytest.mark.parametrize("out_width", [1, 4])
@pytest.mark.parametrize("hidden_layers", [1, 3])
@pytest.mark.parametrize("hidden_width", [1, 5])
@pytest.mark.parametrize("normalization", [LazyBatchNorm1d])
@pytest.mark.parametrize("nonlinearity", [None, torch.nn.ReLU])
def test_low_order_mlp(
    in_width,
    out_width,
    hidden_layers,
    hidden_width,
    normalization,
    nonlinearity,
):

    network = LowOrderMLP(
        in_width=in_width,
        out_width=out_width,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        non_linearity=None if nonlinearity is None else nonlinearity(),
        normalization=normalization,
    )

    batch_size = 4
    x = torch.rand(batch_size, in_width)
    y0 = network(x)
    assert y0.shape[0] == batch_size
    assert y0.shape[1] == out_width
