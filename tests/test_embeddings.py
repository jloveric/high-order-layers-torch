import os

import pytest
import torch

from high_order_layers_torch.positional_embeddings import (
    ClassicSinusoidalEmbedding,
    FourierSeriesEmbedding,
    PiecewiseDiscontinuousPolynomialEmbedding,
    PiecewisePolynomialEmbedding,
)


def test_classic_embedding():

    x = torch.rand([5, 7])
    embedding = ClassicSinusoidalEmbedding(10)
    ans = embedding(x)
    assert ans.shape == torch.Size([5, 7, 10])

    x = torch.rand([5])
    ans = embedding(x)
    assert ans.shape == torch.Size([5, 10])


def test_classic_embedding_throws():
    with pytest.raises(ValueError):
        embedding = ClassicSinusoidalEmbedding(3)
