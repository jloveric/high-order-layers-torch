import logging
from abc import ABC
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d

from high_order_layers_torch.PolynomialLayers import *

from .FunctionalConvolution import Expansion1d, Expansion2d, LagrangeExpand
from .LagrangePolynomial import LagrangeExpand
from .utils import *

logger = logging.getLogger(__name__)


def conv_transpose_wrapper(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    weight_magnitude: float = 1.0,
    rescale_output: bool = False,
    verbose: bool = False,
    convolution: Optional[
        Union[ConvTranspose1d, ConvTranspose2d, ConvTranspose3d]
    ] = ConvTranspose2d,
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
        output_padding=output_padding,
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


class PiecewisePolynomialConvolutionTranspose(nn.Module):
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
        convolution: Union[ConvTranspose1d, ConvTranspose2d, ConvTranspose3d] = None,
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

        self.conv = conv_transpose_wrapper(
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


class PiecewisePolynomialConvolutionTranspose2d(
    PiecewisePolynomialConvolutionTranspose
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
            convolution=ConvTranspose2d,
            expansion_function=PiecewisePolynomialExpand,
            *args,
            **kwargs,
        )
