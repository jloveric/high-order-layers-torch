from .LagrangePolynomial import LagrangeExpand

from functional_layers.PolynomialLayers import *
from torch.nn import Conv2d
import torch.nn as nn
import torch


class ExpansionLayer2D(nn.Module):
    def __init__(self, basis=None):
        super().__init__()
        if basis == None:
            raise Exception(
                'You must define the basis function in ExpansionLayer2D')
        self.basis = basis

    def build(self, input_shape):
        pass

    def call(self, inputs):
        res = self.basis(inputs)
        res = inputs.permute(1, 2, 3, 4, 0)
        res = torch.reshape(
            res, [-1, res.shape[1], res.shape[2], res.shape[3]*res.shape[4]])
        return res


class PolynomialConvolution2D(nn.Module):
    def __init__(self, n: int, in_channels: int, *args, **kwargs):
        super().__init__()
        self.poly = Expansion2d(LagrangeExpand(n))
        self.conv = Conv2D(in_channels=n*in_channels, *args, **kwargs)

    def forward(self, x):
        x = self.poly(x)
        out = self.conv(x)
        return out
