from .LagrangePolynomial import LagrangeExpand
from pytorch_lightning import LightningModule, Trainer

from high_order_layers_torch.PolynomialLayers import *
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
        res = self.basis(
            inputs)  # outputs [basis_size, batches, channels, height, width]
        res = res.permute(1, 3, 4, 2, 0)
        res = torch.reshape(
            res, [res.shape[0], res.shape[1],
                  res.shape[2], res.shape[3]*res.shape[4]]
        )
        res = res.permute(0, 3, 1, 2)
        return res


class FourierConvolution2d(nn.Module):

    def __init__(self, n: int, in_channels: int, kernel_size: int, length: float = 2.0, rescale_output=False, *args, **kwargs):
        super().__init__()
        self.poly = Expansion2d(FourierExpand(n, length))
        self._channels = n*in_channels
        self.conv = Conv2d(in_channels=self._channels,
                           kernel_size=kernel_size, **kwargs)
        self._total_in = self._channels*kernel_size*kernel_size
        self._rescale = 1.0
        if rescale_output is True:
            self._rescale = 1.0/self._total_in

    def forward(self, x):
        x = self.poly(x)
        out = self.conv(x)
        return out/self._rescale


class PolynomialConvolution2d(nn.Module):
    def __init__(self, n: int, in_channels: int, kernel_size: int, segments: int = 1, length: float = 2.0, rescale_output=False, *args, **kwargs):
        """
        TODO: remove "segments" and del from kwargs manually when used.
        Segments is not used in this function, but is included in the parameter list 
        so it isn't passed to Conv2d which does not take arbitrary keywords.
        """
        super().__init__()
        self.poly = Expansion2d(LagrangeExpand(n, length=length))
        self._channels = n*in_channels
        self.conv = Conv2d(in_channels=self._channels,
                           kernel_size=kernel_size, **kwargs)
        self._total_in = self._channels*kernel_size*kernel_size
        self._rescale = 1.0
        if rescale_output is True:
            self._rescale = 1.0/self._total_in

    def forward(self, x):
        x = self.poly(x)
        out = self.conv(x)
        return out/self._rescale


class PiecewisePolynomialConvolution2d(nn.Module):
    def __init__(self, n: int, segments: int,  in_channels: int, kernel_size: int, length: float = 2.0, rescale_output: bool = False, *args, **kwargs):
        super().__init__()
        self.poly = Expansion2d(
            PiecewisePolynomialExpand(n=n, segments=segments, length=length))
        self._channels = ((n-1)*segments+1)*in_channels
        self.conv = Conv2d(in_channels=self._channels,
                           kernel_size=kernel_size, **kwargs)
        self._total_in = self._channels*kernel_size*kernel_size
        self._rescale = 1.0
        if rescale_output is True:
            self._rescale = 1.0/self._total_in

    def forward(self, x):
        x = self.poly(x)
        out = self.conv(x)
        return out/self._rescale


class PiecewiseDiscontinuousPolynomialConvolution2d(nn.Module):
    def __init__(self, n: int, segments: int,  in_channels: int, kernel_size: int, length: float = 2.0, rescale_output: bool = True, *args, **kwargs):
        super().__init__()
        self.poly = Expansion2d(
            PiecewiseDiscontinuousPolynomialExpand(n=n, segments=segments, length=length))
        self._channels = n*segments*in_channels
        self.conv = Conv2d(in_channels=self._channels,
                           kernel_size=kernel_size, **kwargs)
        self._total_in = self._channels*kernel_size*kernel_size
        self._rescale = 1.0
        if rescale_output is True:
            self._rescale = 1.0/self._total_in

    def forward(self, x):
        x = self.poly(x)
        out = self.conv(x)
        return out/self._rescale
