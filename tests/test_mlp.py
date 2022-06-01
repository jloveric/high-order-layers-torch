import os
import pytest
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.PolynomialLayers import *
from high_order_layers_torch.networks import *
from torch.nn import LazyBatchNorm1d, LazyInstanceNorm1d


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("in_width", [5])
@pytest.mark.parametrize("out_width", [1, 4])
@pytest.mark.parametrize("hidden_layers", [1, 3])
@pytest.mark.parametrize("hidden_width", [1, 5])
@pytest.mark.parametrize("n0", [2, 3])
@pytest.mark.parametrize("normalization", [LazyBatchNorm1d])
def test_interpolate_mlp(
    segments, in_width, out_width, hidden_layers, hidden_width, n0, normalization
):

    network = HighOrderMLP(
        layer_type="continuous",
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
        normalization=normalization(),
    )

    batch_size = 4
    x = torch.rand(batch_size, in_width)
    y0 = network(x)
    assert y0.shape[0] == batch_size
    assert y0.shape[1] == out_width
