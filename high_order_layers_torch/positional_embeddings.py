import math

import torch
from torch import nn

from high_order_layers_torch.PolynomialLayers import (
    FourierSeries,
    PiecewiseDiscontinuousPolynomial,
    PiecewisePolynomial,
)


class ClassicSinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Traditional positional embedding.
        Args :
            dim : The dimension of the embedding.  This value needs to be even as
            it will be split half and half into sin and cos components.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension needs to be even, got {dim}.")
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


"""
The Fourier, PiecwisePolynomial and PiecewiseDiscontinuousPolynomial embeddings are just their
respective layers.  I've described them as embeddings to make their intent clear.
"""


def FourierSeriesEmbedding(
    n: int, in_features: int, out_features: int, length: float = 2.0, **kwargs
):
    """
    High order embedding using fourier layer
    """
    FourierSeries(
        n=n, in_features=in_features, out_features=out_features, length=length
    )


def PiecewisePolynomialEmbedding(
    n: int,
    in_features: int,
    out_features: int,
    segments: int,
    length=2.0,
    weight_magnitude=1.0,
    periodicity: float = None,
    **kwargs,
):
    """
    High order embedding using piecewise polynomial layer
    """
    return PiecewisePolynomial(
        n=n,
        in_features=in_features,
        out_features=out_features,
        segments=segments,
        length=length,
        weight_magnitude=weight_magnitude,
        periodicity=periodicity,
    )


def PiecewiseDiscontinuousPolynomialEmbedding(
    n: int,
    in_features: int,
    out_features: int,
    segments: int,
    length=2.0,
    weight_magnitude=1.0,
    periodicity: float = None,
    **kwargs,
):
    """
    High order embedding using sparse discontinuous piecewise polynomial
    """
    return PiecewiseDiscontinuousPolynomial(
        n=n,
        in_features=in_features,
        out_features=out_features,
        segments=segments,
        length=length,
        weight_magnitude=weight_magnitude,
        periodicity=periodicity,
    )
