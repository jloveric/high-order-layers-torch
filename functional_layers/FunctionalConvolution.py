from .LagrangePolynomial import LagrangeExpand
from pytorch_lightning import LightningModule, Trainer

from functional_layers.PolynomialLayers import *
from torch.nn import Conv2d
import torch.nn as nn
import torch


class Expansion2d(nn.Module):
    def __init__(self, basis=None):
        super().__init__()
        if basis == None:
            raise Exception(
                'You must define the basis function in ExpansionLayer2D')
        self.basis = basis

    def build(self, input_shape):
        pass

    def __call__(self, inputs):
        """
        Expand input
        Args :
            inputs : Tensor of shape [batches, channels, height, width]
        Return :
            Tensor of shape [batches, channels*(basis size), height, width]
        """
        res = self.basis(inputs) # outputs [basis_size, batches, channels, height, width]
        res = res.permute(1, 3, 4, 2, 0)
        res = torch.reshape(
            res, [res.shape[0], res.shape[1],
                  res.shape[2], res.shape[3]*res.shape[4]]
        )
        res = res.permute(0, 3, 1, 2)
        return res


class FourierConvolution2d(nn.Module):

    def __init__(self, n: int, in_channels: int, *args, **kwargs):
        super().__init__()
        self.poly = Expansion2d(FourierExpand(n))
        self.conv = Conv2d(in_channels=n*in_channels, **kwargs)

    def forward(self, x):
        x = self.poly(x)
        out = self.conv(x)
        return out


class PolynomialConvolution2d(nn.Module):
    def __init__(self, n: int, in_channels: int, *args, **kwargs):
        super().__init__()
        self.poly = Expansion2d(LagrangeExpand(n))
        self.conv = Conv2d(in_channels=n*in_channels, **kwargs)

    def forward(self, x):
        x = self.poly(x)
        out = self.conv(x)
        return out


class PiecewisePolynomialConvolution2d(nn.Module):
    def __init__(self, n: int, segments: int,  in_channels: int, *args, **kwargs):
        super().__init__()
        self.poly = Expansion2d(PiecewisePolynomialExpand(n, segments))
        channels = ((n-1)*segments+1)*in_channels
        #print('channels', channels)
        self.conv = Conv2d(in_channels=channels, **kwargs)

    def forward(self, x):
        #print('x after shape conv 1', x.shape)
        x = self.poly(x)
        #print('x after.shape conv 2', x.shape)
        out = self.conv(x)
        #print('x after.shape conv 3', x.shape)
        return out
