import torch.nn as nn
from torch.nn import Linear

from .FunctionalConvolution import *
from .FunctionalConvolutionTranspose import *
from .PolynomialLayers import *
from .ProductLayer import *
from .utils import max_abs_normalization, l2_normalization, max_abs_normalization_nd


class MaxAbsNormalization(nn.Module):
    """
    Normalization for the 1D case (MLP)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return max_abs_normalization(x, eps=self._eps)


class MaxAbsNormalizationND(nn.Module):
    """
    Normalization for ND case, specifically convolutions
    but also works for MLP layers.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return max_abs_normalization_nd(x, eps=self._eps)


class L2Normalization(nn.Module):
    """
    L2 normalization for MLP layers
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return l2_normalization(x, eps=self._eps)


normalization_layers = {
    "max_abs": MaxAbsNormalization,
    "l2": L2Normalization,
}


def LinearAdapter(*args, in_features: int, out_features: int, **kwargs):
    return Linear(in_features=in_features, out_features=out_features, bias=True)


class LinearReluAdapter(nn.Module):
    def __init__(self, *args, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.f = Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x):
        return torch.nn.functional.relu(self.f.forward(x))


fc_layers = {
    "baseline_relu": LinearReluAdapter,  # Linear layer folowed by relu
    "baseline": LinearAdapter,  # Standard linear layer
    "continuous": PiecewisePolynomial,
    "continuous_prod": PiecewisePolynomialProd,
    "discontinuous": PiecewiseDiscontinuousPolynomial,
    "discontinuous_prod": PiecewiseDiscontinuousPolynomialProd,
    "polynomial": Polynomial,
    "polynomial_prod": PolynomialProd,
    "product": Product,
    "fourier": FourierSeries,
}

convolutional_layers = {
    "continuous2d": PiecewisePolynomialConvolution2d,
    "continuous1d": PiecewisePolynomialConvolution1d,
    "continuous_prod2d": None,  # PiecewisePolynomialProd,
    "discontinuous2d": PiecewiseDiscontinuousPolynomialConvolution2d,
    "discontinuous1d": PiecewiseDiscontinuousPolynomialConvolution1d,
    "discontinuous_prod2d": None,  # PiecewiseDiscontinuousPolynomialProd,
    "polynomial2d": PolynomialConvolution2d,
    "polynomial1d": PolynomialConvolution1d,
    "polynomial_prod2d": None,  # PolynomialConvolutionProd2d,
    "product2d": None,  # ProductConvolution2d,
    "fourier2d": FourierConvolution2d,
    "fourier1d": FourierConvolution1d,
}

convolutional_transpose_layers = {
    "continuous2d": PiecewisePolynomialConvolutionTranspose2d
}


def high_order_fc_layers(layer_type: str, **kwargs):

    if layer_type in fc_layers.keys():
        return fc_layers[layer_type](**kwargs)

    raise ValueError(
        f"Fully connected layer type {layer_type} not recognized.  Must be one of {list(fc_layers.keys())}"
    )


def high_order_convolution_layers(layer_type: str, **kwargs):

    if layer_type in convolutional_layers.keys():
        return convolutional_layers[layer_type](**kwargs)

    raise ValueError(
        f"Convolutional layer type {layer_type} not recognized.  Must be one of {list(convolutional_layers.keys())}"
    )


def high_order_convolution_transpose_layers(layer_type: str, **kwargs):

    if layer_type in convolutional_transpose_layers.keys():
        return convolutional_transpose_layers[layer_type](**kwargs)

    raise ValueError(
        f"ConvolutionalTranspose layer type {layer_type} not recognized.  Must be one of {list(convolutional_transpose_layers.keys())}"
    )
