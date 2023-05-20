import os

import pytest

from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.networks import *
from high_order_layers_torch.PolynomialLayers import *
import torch

torch.set_default_device(device="cpu")


@pytest.mark.parametrize(
    "n_in,n_out,in_features,out_features,segments",
    [(3, 5, 3, 2, 5), (5, 5, 2, 3, 2), (7, 5, 3, 2, 5)],
)
@pytest.mark.parametrize(
    "layer_type", [PiecewisePolynomial, PiecewiseDiscontinuousPolynomial]
)
def test_interpolate_layer(
    n_in: int,
    n_out: int,
    in_features: int,
    out_features: int,
    segments: int,
    layer_type: Union[PiecewisePolynomial, PiecewiseDiscontinuousPolynomial],
):
    layer_in = layer_type(
        n=n_in, in_features=in_features, out_features=out_features, segments=segments
    )
    layer_out = layer_type(
        n=n_out, in_features=in_features, out_features=out_features, segments=segments
    )
    interpolate_polynomial_layer(layer_in=layer_in, layer_out=layer_out)

    x_in = torch.rand(2, in_features)
    x_out_start = layer_in(x_in)
    x_out_end = layer_out(x_in)

    if n_in <= n_out:  # There should be no loss of information
        assert torch.allclose(x_out_start, x_out_end, rtol=1e-5)
    else:
        pass


@pytest.mark.parametrize(
    "segments,in_width,out_width,hidden_layers,hidden_width,n0,n1",
    [(2, 5, 5, 2, 5, 2, 3), (2, 5, 3, 3, 3, 3, 5)],
)
@pytest.mark.parametrize(
    "layer_type",
    ["continuous", "discontinuous", "switch_continuous", "switch_discontinuous"],
)
def test_interpolate_mlp(
    segments, in_width, out_width, hidden_layers, hidden_width, n0, n1, layer_type
):
    network_in = HighOrderMLP(
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
    )
    network_out = HighOrderMLP(
        layer_type=layer_type,
        n=n1,
        in_width=in_width,
        out_width=out_width,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        n_in=n1,
        n_out=n1,
        n_hidden=n1,
        in_segments=segments,
        out_segments=segments,
        hidden_segments=segments,
    )

    interpolate_high_order_mlp(network_in, network_out)

    x = torch.rand(2, 5)
    y0 = network_in(x)
    y1 = network_out(x)
    assert torch.allclose(y0, y1, rtol=1e-4)


@pytest.mark.parametrize(
    "segments_in,segments_out,in_features,out_features,n",
    [(3, 9, 3, 2, 3), (5, 5, 2, 3, 2), (7, 5, 3, 2, 2), (2, 12, 3, 2, 6)],
)
@pytest.mark.parametrize("layer_type", [PiecewisePolynomial])
def test_refine_polynomial_layer(
    segments_in: int,
    segments_out: int,
    in_features: int,
    out_features: int,
    n: int,
    layer_type: Union[PiecewisePolynomial, PiecewiseDiscontinuousPolynomial],
):
    layer_in = layer_type(
        n=n, in_features=in_features, out_features=out_features, segments=segments_in
    )
    layer_out = layer_type(
        n=n, in_features=in_features, out_features=out_features, segments=segments_out
    )
    refine_polynomial_layer(layer_in=layer_in, layer_out=layer_out)

    x_in = torch.rand(2, in_features)
    x_out_start = layer_in(x_in)
    x_out_end = layer_out(x_in)

    if segments_in <= segments_out:  # There should be no loss of information
        assert torch.allclose(x_out_start, x_out_end, rtol=1e-3)
    else:
        pass


@pytest.mark.parametrize(
    "segments_in,segments_out,in_width,out_width,hidden_layers,hidden_width",
    [(2, 4, 5, 5, 2, 5), (3, 6, 5, 3, 3, 3)],
)
@pytest.mark.parametrize("n_in", [2, 3])
@pytest.mark.parametrize("n_out", [3, 4])
@pytest.mark.parametrize(
    "layer_type", ["continuous", "switch_continuous"]
)  # TODO: add back in "discontinuous" when working
def test_h_refinement_of_mlp(
    segments_in,
    segments_out,
    in_width,
    out_width,
    hidden_layers,
    hidden_width,
    n_in,
    n_out,
    layer_type,
):
    network_in = HighOrderMLP(
        layer_type=layer_type,
        n=n_in,
        in_width=in_width,
        out_width=out_width,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        n_in=n_in,
        n_hidden=n_in,
        in_segments=segments_in,
        out_segments=segments_in,
        hidden_segments=segments_in,
        device="cpu",
    )
    network_out = HighOrderMLP(
        layer_type=layer_type,
        n=n_out,
        in_width=in_width,
        out_width=out_width,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        n_in=n_out,
        n_out=n_out,
        n_hidden=n_out,
        in_segments=segments_out,
        out_segments=segments_out,
        hidden_segments=segments_out,
        device="cpu",
    )

    hp_refine_high_order_mlp(network_in, network_out)

    x = torch.rand(2, 5, device="cpu")
    y0 = network_in(x)
    y1 = network_out(x)
    assert torch.allclose(y0, y1, rtol=1e-3)
