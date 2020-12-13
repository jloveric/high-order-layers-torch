import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.metrics.functional import accuracy
from .PolynomialLayers import *
from .ProductLayer import *
from .FunctionalConvolution import *

"""
def linear_wrapper(in_features: int, out_features: int, bias: bool = True, **kwargs):
    return nn.Linear(in_features, out_features, bias)
"""

fc_layers = {
    "continuous": PiecewisePolynomial,
    "continuous_prod": PiecewisePolynomialProd,
    "discontinuous": PiecewiseDiscontinuousPolynomial,
    "discontinuous_prod": PiecewiseDiscontinuousPolynomialProd,
    "polynomial": Polynomial,
    "polynomial_prod": PolynomialProd,
    "product": Product,
    "fourier": FourierSeries
}

convolutional_layers = {
    "continuous": PiecewisePolynomialConvolution2d,
    "continuous_prod": None,  # PiecewisePolynomialProd,
    "discontinuous": PiecewiseDiscontinuousPolynomialConvolution2d,
    "discontinuous_prod": None,  # PiecewiseDiscontinuousPolynomialProd,
    "polynomial": PolynomialConvolution2d,
    "polynomial_prod": None,  # PolynomialConvolutionProd2d,
    "product": None,  # ProductConvolution2d,
    "fourier": FourierConvolution2d
}


def high_order_fc_layers(layer_type: str, **kwargs):

    if layer_type in fc_layers.keys():
        return fc_layers[layer_type](**kwargs)

    raise ValueError(
        f"Fully connected layer type {layer_type} not recognized.  Must be one of {list(fc_layers.keys())}")


def high_order_convolution_layers(layer_type: str, **kwargs):

    if layer_type in convolutional_layers.keys():
        return convolutional_layers[layer_type](**kwargs)

    raise ValueError(
        f"Convolutional layer type {layer_type} not recognized.  Must be one of {list(convolutional_layers.keys())}")
