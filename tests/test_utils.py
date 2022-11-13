import os

import pytest
import torch

from high_order_layers_torch.utils import make_periodic, positions_from_mesh


def test_positions_from_mesh():
    width = 10
    height = 10
    rotations = 2
    positions = positions_from_mesh(
        width=width, height=height, rotations=rotations, normalize=True, device="cpu"
    )

    assert len(positions) == 4
    for i in range(len(positions)):
        assert torch.all(positions[i] >= -1.0).item() is True
        assert torch.all(positions[i] <= 1.0).item() is True

    positions = positions_from_mesh(
        width=width, height=height, rotations=rotations, normalize=False, device="cpu"
    )
    assert len(positions) == 4
    for i in range(len(positions)):
        assert torch.all(positions[i] >= -5.0).item() is True
        assert torch.all(positions[i] <= 5.0).item() is True
