import os

import pytest

from high_order_layers_torch.FunctionalConvolution import *
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.LagrangePolynomial import LagrangePoly
import torch.nn.functional as F
from high_order_layers_torch.layers import (
    L2Normalization,
    MaxAbsNormalization,
    MaxAbsNormalizationND,
    MaxAbsNormalizationLast,
    MaxCenterNormalization,
    SwitchLayer,
    fixed_rotation_layer,
    initialize_polynomial_layer,
)
from high_order_layers_torch.sparse_optimizers import SparseLion
from high_order_layers_torch.PolynomialLayers import PiecewiseDiscontinuousPolynomial
import torch


def test_sparse_lion():
    """
    Make sure only a subset of the values are being used in a standard
    weight update
    """

    layer = PiecewiseDiscontinuousPolynomial(
        n=2, in_features=1, out_features=1, segments=10
    )
    optim = SparseLion(params=layer.parameters(), lr=1e-4)

    # Activate only on value
    x = torch.ones(1, 1) * 0.5
    target = torch.ones(1, 1) * 0.2
    out = layer(x)

    loss = F.mse_loss(out, target)
    loss.backward()

    evaluated_grads = layer.w.grad[layer.w.grad != 0.0]
    print("evaluated_grads", evaluated_grads)

    assert evaluated_grads.shape == torch.Size([2])
    assert layer.w.grad.shape == torch.Size([1, 1, 10 * 2])

    print("weight gradients", layer.w.grad)
    w_before = layer.w.clone()
    optim.step()
    w_after = layer.w.clone()
    assert torch.any(torch.abs(w_after - w_before) > 0)

    out_final = layer(x)
    diff = torch.abs(out_final - out)
    assert torch.any(diff > 0)
    print("diff", diff)
