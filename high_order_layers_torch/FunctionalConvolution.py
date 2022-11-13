import logging
from abc import ABC
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Conv3d

from high_order_layers_torch.PolynomialLayers import *

from .LagrangePolynomial import LagrangeExpand
from .utils import *

logger = logging.getLogger(__name__)


def conv_wrapper(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    weight_magnitude: float = 1.0,
    rescale_output: bool = False,
    verbose: bool = False,
    convolution: Optional[Union[Conv1d, Conv2d, Conv3d]] = Conv2d,
    **kwargs,
):
    """
    Inputs need to be an exact clone of those in torch conv2d including
    defaults.  Function allows you to pass extra arguments without breaking
    conv2d.
    """

    conv = convolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        # Bias should always be false as the bias is already included in these methods.
        bias=False,
        padding_mode=padding_mode,
    )
    in_features = in_channels * kernel_size * kernel_size

    if verbose is True:
        logger.info(f"in_channels {in_channels} out_channels {out_channels}")
        logger.info(f"conv.weight.shape {conv.weight.shape}")

    # We don't want to use the standard conv initialization
    # since this is a bit different.
    if rescale_output is False:
        conv.weight.data.uniform_(
            -weight_magnitude / in_features, weight_magnitude / in_features
        )
    elif rescale_output is True:
        conv.weight.data.uniform_(-weight_magnitude, weight_magnitude)
    else:
        logger.info("Using kaiming for weight initialization")

    return conv


# TODO: Pretty sure this doesn't need to be an nn.Module
class Expansion2d(nn.Module):
    def __init__(self, basis: Callable[[Tensor, int], float] = None):
        """
        Expand an input by a function defined by basis.

        Args :
            - basis: function to expand input by.
        """
        super().__init__()
        if basis == None:
            raise Exception("You must define the basis function in ExpansionLayer2D")
        self.basis = basis

    def build(self, input_shape):
        pass

    def __call__(self, inputs: Tensor) -> Tensor:
        """
        Expand input
        Args :
            inputs : Tensor of shape [batches, channels, height, width]
        Return :
            Tensor of shape [batches, channels*(basis size), height, width]
        """
        res = self.basis(
            inputs
        )  # outputs [basis_size, batches, channels, height, width]
        res = res.permute(1, 3, 4, 2, 0)
        res = torch.reshape(
            res, [res.shape[0], res.shape[1], res.shape[2], res.shape[3] * res.shape[4]]
        )
        res = res.permute(0, 3, 1, 2)
        return res


class Expansion1d(nn.Module):
    def __init__(self, basis: Callable[[Tensor, int], float] = None):
        """
        Expand an input by a function defined by basis.

        Args :
            - basis: function to expand input by.
        """
        super().__init__()
        if basis == None:
            raise Exception("You must define the basis function in ExpansionLayer2D")
        self.basis = basis

    def build(self, input_shape):
        pass

    def __call__(self, inputs: Tensor) -> Tensor:
        """
        Expand input
        Args :
            inputs : Tensor of shape [batches, channels, width]
        Return :
            Tensor of shape [batches, channels*(basis size), width]
        """
        res = self.basis(inputs)  # outputs [basis_size, batches, channels, width]
        res = res.permute(1, 3, 2, 0)
        res = torch.reshape(
            res, [res.shape[0], res.shape[1], res.shape[2] * res.shape[3]]
        )
        res = res.permute(0, 2, 1)  # batches, basis_size*channels, width
        return res


# TODO: redo all these with mixins (instead of specialization) since it will be more compact.
class FourierConvolution(nn.Module):
    def __init__(
        self,
        n: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output=False,
        expansion: Union[Expansion1d, Expansion2d] = None,
        convolution: Union[Conv1d, Conv2d, Conv3d] = None,
        *args,
        **kwargs,
    ):
        """
        Fourier series convolutional layer.

        Args :
            - n : number of fourier series components. n=1 is a constant, n=3 contains both first sin an consine components.
            - in_channels : number of input channels
            - kernel_size : size of the kernel
            - length : Range of the polynomial interpolation points.  length = 2 implies [-1, 1] so the interpolation points
                are in that range.  Anything outside that range could grow.
            - rescale_output: If rescale output is True then the output is divided by the number of inputs for each output,
                in effect taking the average.  This is generally not necessary for the fourier series.
        """
        super().__init__()
        self.poly = expansion(FourierExpand(n, length))
        self._channels = n * in_channels
        self.conv = conv_wrapper(
            in_channels=self._channels,
            kernel_size=kernel_size,
            convolution=convolution,
            **kwargs,
        )
        self._total_in = in_channels * kernel_size * kernel_size
        self._rescale = 1.0
        if rescale_output is True:
            self._rescale = 1.0 / self._total_in

    def forward(self, x: Tensor) -> Tensor:
        x = self.poly(x)
        out = self.conv(x)
        return out * self._rescale


class FourierConvolution2d(FourierConvolution):
    def __init__(
        self,
        n: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            n=n,
            in_channels=in_channels,
            kernel_size=kernel_size,
            length=length,
            rescale_output=rescale_output,
            expansion=Expansion2d,
            convolution=Conv2d,
            *args,
            **kwargs,
        )


class FourierConvolution1d(FourierConvolution):
    def __init__(
        self,
        n: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            n=n,
            in_channels=in_channels,
            kernel_size=kernel_size,
            length=length,
            rescale_output=rescale_output,
            conv_wrapper=Conv1d,
            *args,
            **kwargs,
        )


class PolynomialConvolution(nn.Module):
    def __init__(
        self,
        n: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output=False,
        periodicity: float = None,
        expansion: Union[Expansion1d, Expansion2d] = None,
        convolution: Union[Conv1d, Conv2d, Conv3d] = None,
        *args,
        **kwargs,
    ):
        """
        Polynomial convolutional layer.

        Args :
            - n : number of weights or nodes.  Polynomial order is n-1 so quadratic would be n=3.
            - in_channels : number of input channels
            - kernel_size : size of the kernel
            - length : Range of the polynomial interpolation points.  length = 2 implies [-1, 1] so the interpolation points
                are in that range.  Anything outside that range could grow.
            - rescale_output: If rescale output is True then the output is divided by the number of inputs for each output,
                in effect taking the average.
        """
        super().__init__()
        self.poly = expansion(LagrangeExpand(n, length=length))
        self._channels = n * in_channels
        self.periodicity = periodicity
        self.conv = conv_wrapper(
            in_channels=self._channels,
            kernel_size=kernel_size,
            convolution=convolution,
            **kwargs,
        )
        self._total_in = in_channels * kernel_size * kernel_size
        self._rescale = 1.0
        if rescale_output is True:
            self._rescale = 1.0 / self._total_in

    def forward(self, x: Tensor) -> Tensor:
        periodicity = self.periodicity
        if periodicity is not None:
            x = make_periodic(x, periodicity)
        x = self.poly(x)
        out = self.conv(x)
        return out * self._rescale


class PolynomialConvolution2d(PolynomialConvolution):
    def __init__(
        self,
        n: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output=False,
        periodicity: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            n=n,
            in_channels=in_channels,
            kernel_size=kernel_size,
            length=length,
            rescale_output=rescale_output,
            periodicity=periodicity,
            expansion=Expansion2d,
            convolution=Conv2d,
            *args,
            **kwargs,
        )


# TODO: Test this works!
class PolynomialConvolution1d(PolynomialConvolution):
    def __init__(
        self,
        n: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output=False,
        periodicity: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            n=n,
            in_channels=in_channels,
            kernel_size=kernel_size,
            length=length,
            rescale_output=rescale_output,
            periodicity=periodicity,
            expansion=Expansion1d,
            convolution=Conv1d,
            *args,
            **kwargs,
        )


class PiecewisePolynomialConvolution(nn.Module):
    def __init__(
        self,
        n: int,
        segments: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output: bool = False,
        periodicity: float = None,
        expansion: Union[Expansion1d, Expansion2d] = None,
        convolution: Union[Conv1d, Conv2d, Conv3d] = None,
        expansion_function: Any = None,
        *args,
        **kwargs,
    ):
        """
        Piecewise continuous polynomial convolutional layer.  The boundary between each polynomial are continuous.

        Args :
            - n : number of weights or nodes.  Polynomial order is n-1 so quadratic would be n=3.
            - segments: The number of segments in the piecewise polynomial.
            - in_channels : number of input channels
            - kernel_size : size of the kernel
            - length : Range of the piecewise polynomial interpolation points.  length = 2 implies [-1, 1] so the interpolation points
                are in that range.
            - rescale_output: If rescale output is True then the output is divided by the number of inputs for each output,
                in effect taking the average.
        """
        super().__init__()
        self.poly = expansion(expansion_function(n=n, segments=segments, length=length))
        self._channels = ((n - 1) * segments + 1) * in_channels
        self._out_channels = out_channels
        self.periodicity = periodicity

        self.conv = conv_wrapper(
            in_channels=self._channels,
            out_channels=self._out_channels,
            kernel_size=kernel_size,
            convolution=convolution,
            **kwargs,
        )
        self._total_in = in_channels * kernel_size * kernel_size
        self._rescale = 1.0
        if rescale_output is True:
            self._rescale = 1.0 / self._total_in

    def forward(self, x: Tensor) -> Tensor:
        periodicity = self.periodicity
        if periodicity is not None:
            x = make_periodic(x, periodicity)
        x = self.poly(x)
        out = self.conv(x)
        return out * self._rescale


class PiecewisePolynomialConvolution2d(PiecewisePolynomialConvolution):
    def __init__(
        self,
        n: int,
        segments: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output: bool = False,
        periodicity: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            n=n,
            segments=segments,
            in_channels=in_channels,
            kernel_size=kernel_size,
            length=length,
            rescale_output=rescale_output,
            periodicity=periodicity,
            expansion=Expansion2d,
            convolution=Conv2d,
            expansion_function=PiecewisePolynomialExpand,
            *args,
            **kwargs,
        )


class PiecewisePolynomialConvolution1d(PiecewisePolynomialConvolution):
    def __init__(
        self,
        n: int,
        segments: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output: bool = False,
        periodicity: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            n=n,
            segments=segments,
            in_channels=in_channels,
            kernel_size=kernel_size,
            length=length,
            rescale_output=rescale_output,
            periodicity=periodicity,
            expansion=Expansion1d,
            convolution=Conv1d,
            expansion_function=PiecewisePolynomialExpand1d,
            *args,
            **kwargs,
        )


class PiecewiseDiscontinuousPolynomialConvolution(nn.Module):
    def __init__(
        self,
        n: int,
        segments: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output: bool = False,
        periodicity: float = None,
        expansion: Union[Expansion1d, Expansion2d] = None,
        convolution: Union[Conv1d, Conv2d, Conv3d] = None,
        expansion_function: Any = None,
        *args,
        **kwargs,
    ):
        """
        Discontinuous piecewise polynomial convolutional layer.  The boundary between each polynomial can be discontinuous.
        Args :
            - n : number of weights or nodes.  Polynomial order is n-1 so quadratic would be n=3.
            - segments: The number of segments in the piecewise polynomial.
            - in_channels : number of input channels
            - kernel_size : size of the kernel
            - length : Range of the piecewise polynomial interpolation points.  length = 2 implies [-1, 1] so the interpolation points
                are in that range.
            - rescale_output: If rescale output is True then the output is divided by the number of inputs for each output,
                in effect taking the average.
        """
        super().__init__()
        self.poly = expansion(expansion_function(n=n, segments=segments, length=length))
        self._channels = n * segments * in_channels
        self.periodicity = periodicity
        self.conv = conv_wrapper(
            in_channels=self._channels,
            kernel_size=kernel_size,
            convolution=convolution,
            **kwargs,
        )
        self._total_in = in_channels * kernel_size * kernel_size
        self._rescale = 1.0
        if rescale_output is True:
            self._rescale = 1.0 / self._total_in

    def forward(self, x: Tensor) -> Tensor:
        periodicity = self.periodicity
        if periodicity is not None:
            x = make_periodic(x, periodicity)
        x = self.poly(x)
        out = self.conv(x)
        return out * self._rescale


class PiecewiseDiscontinuousPolynomialConvolution2d(
    PiecewiseDiscontinuousPolynomialConvolution
):
    def __init__(
        self,
        n: int,
        segments: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output: bool = False,
        periodicity: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            n=n,
            segments=segments,
            in_channels=in_channels,
            kernel_size=kernel_size,
            length=length,
            rescale_output=rescale_output,
            periodicity=periodicity,
            expansion=Expansion2d,
            convolution=Conv2d,
            expansion_function=PiecewiseDiscontinuousPolynomialExpand,
            *args,
            **kwargs,
        )


class PiecewiseDiscontinuousPolynomialConvolution1d(
    PiecewiseDiscontinuousPolynomialConvolution
):
    def __init__(
        self,
        n: int,
        segments: int,
        in_channels: int,
        kernel_size: int,
        length: float = 2.0,
        rescale_output: bool = False,
        periodicity: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            n=n,
            segments=segments,
            in_channels=in_channels,
            kernel_size=kernel_size,
            length=length,
            rescale_output=rescale_output,
            periodicity=periodicity,
            expansion=Expansion1d,
            convolution=Conv1d,
            expansion_function=PiecewiseDiscontinuousPolynomialExpand1d,
            *args,
            **kwargs,
        )
