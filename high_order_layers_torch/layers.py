import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear

from .FunctionalConvolution import *
from .FunctionalConvolutionTranspose import *
from .PolynomialLayers import *
from .ProductLayer import *


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
