import math
from typing import List

import torch
from torch import Tensor


def max_abs(x: Tensor, dim: int = 1):
    return torch.max(x.abs(), dim=dim, keepdim=True)[0]


def max_abs_normalization(x: Tensor, eps: float = 1e-6, dim: int = 1):
    return x / (max_abs(x, dim=dim) + eps)


def max_abs_normalization_last(x: Tensor, eps: float = 1e-6):
    return x / (max_abs(x, dim=len(x.shape) - 1) + eps)


def max_center_normalization(x: Tensor, eps: float = 1e-6, dim: int = 1):
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    min_x = torch.min(x, dim=dim, keepdim=True)[0]

    midrange = 0.5 * (max_x + min_x)
    mag = max_x - midrange

    centered = x - midrange
    return centered / (mag + eps)


def max_center_normalization_last(x: Tensor, eps: float = 1e-6):
    max_x = torch.max(x, dim=len(x.shape) - 1, keepdim=True)[0]
    min_x = torch.min(x, dim=len(x.shape) - 1, keepdim=True)[0]

    midrange = 0.5 * (max_x + min_x)
    mag = max_x - midrange

    centered = x - midrange
    return centered / (mag + eps)


def l2_normalization(x: Tensor, eps: float = 1e-6):
    return x / (x.norm(2, 1, keepdim=True) + eps)


def max_abs_normalization_nd(x: Tensor, eps: float = 1e-6):
    shape = x.shape
    xn = x.reshape(shape[0], -1)
    norm = xn / (max_abs(xn) + eps)
    return norm.reshape(shape)


norm_type = {
    "max_abs": max_abs_normalization,
    "l2": l2_normalization,
}


def make_periodic(x, periodicity: float):
    xp = x + 0.5 * periodicity
    xp = torch.remainder(xp, 2 * periodicity)  # always positive
    xp = torch.where(xp > periodicity, 2 * periodicity - xp, xp)
    xp = xp - 0.5 * periodicity
    return xp


def positions_from_mesh(
    width: int, height: int, rotations: int = 1, device="cuda", normalize: bool = False
) -> List[Tensor]:
    """
    Produce a list of 2d "positions" from a mesh grid.  Assign a 1D value to each point
    in the grid for example x, or y or the diagonal x+y... The outputs here can then be
    used for positional encoding.

    Args :
        width : The width of the array in elements
        height : The height of the array in elements
        rotations : The number of rotations to return.
            1 returns standard x, y (2 coordinate axis)
            2 returns x, y, x+y, x-y (4 coordinate axis)
            3 returns 6 coordinate axes (3 axes and the axis orthogonal to each)
        normalize : If true the range will be normalized to [-1, 1] otherwise
        it goes from [-max_dim//2, max_dim//2-1]
    Returns :
        A list of meshes and the rotated axis
    """
    max_dim = max(width, height)
    scale = max_dim / (max_dim - 1)

    xv, yv = torch.meshgrid([torch.arange(width), torch.arange(height)])
    xv = xv.to(device=device)
    yv = yv.to(device=device)
    xv = scale * xv
    yv = scale * yv

    # Coordinate values range from
    line_list = []
    for i in range(rotations):
        theta = (math.pi / 2.0) * (i / rotations)
        rot_x = math.cos(theta)
        rot_y = math.sin(theta)

        # Add the line and the line orthogonal
        r1 = rot_x * xv + rot_y * yv
        r1_max = torch.max(r1)
        r1_min = torch.min(r1)
        dr1 = r1_max - r1_min

        r2 = rot_x * xv - rot_y * yv
        r2_max = torch.max(r2)
        r2_min = torch.min(r2)
        dr2 = r2_max - r2_min

        # Rescale these so they have length segments
        # and are centered at (0,0)
        r1 = ((r1 - r1_min) / dr1 - 0.5) * max_dim
        r2 = ((r2 - r2_min) / dr2 - 0.5) * max_dim

        # We want the values to range from -1 to 1
        if normalize is True:
            r1 = 2 * (r1 + max_dim / 2) / max_dim - 1
            r2 = 2 * (r2 + max_dim / 2) / max_dim - 1

        line_list.append(r1)
        line_list.append(r2)

    return line_list
