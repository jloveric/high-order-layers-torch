import torch.nn as nn
from torch import Tensor
from high_order_layers_torch.layers import (
    high_order_convolution_layers,
    high_order_fc_layers,
)
from typing import Any, Callable, List, Union
from high_order_layers_torch.PolynomialLayers import interpolate_polynomial_layer


class HighOrderMLP(nn.Module):
    def __init__(
        self,
        layer_type: str,
        n: str,
        in_width: int,
        out_width: int,
        hidden_layers: int,
        hidden_width: int,
        scale: float = 2.0,
        n_in: int = None,
        n_out: int = None,
        n_hidden: int = None,
        rescale_output: bool = False,
        periodicity: float = None,
        non_linearity=None,
        in_segments: int = None,
        out_segments: int = None,
        hidden_segments: int = None,
        normalization: Callable[[Any], Tensor] = None,
    ) -> None:
        """
        Args :
            layer_type: Type of layer
                "continuous", "discontinuous",
                "polynomial", "fourier",
                "product", "continuous_prod",
                "discontinuous_prod"
            n:  Base number of nodes (or fourier components).  If none of the others are set
                then this value is used.
            in_width: Input width.
            out_width: Output width
            hidden_layers: Number of hidden layers.
            hidden_width: Number of hidden units
            scale: Scale of the segments.  A value of 2 would be length 2 (or period 2)
            n_in: Number of input nodes for interpolation or fourier components.
            n_out: Number of output nodes for interpolation or fourier components.
            n_hidden: Number of hidden nodes for interpolation or fourier components.
            rescale_output: Whether to average the outputs
            periodicity: Whether to make polynomials periodic after given length.
            non_linearity: Whether to apply a nonlinearity after each layer (except output)
            in_segments: Number of input segments for each link.
            out_segments: Number of output segments for each link.
            hidden_segments: Number of hidden segments for each link.
            normalization: Normalization to apply after each layer (before any additional nonlinearity).
        """
        super().__init__()
        layer_list = []
        n_in = n_in or n
        n_hidden = n_hidden or n
        n_out = n_out or n

        input_layer = high_order_fc_layers(
            layer_type=layer_type,
            n=n_in,
            in_features=in_width,
            out_features=hidden_width,
            segments=in_segments,
            rescale_output=rescale_output,
            scale=scale,
            periodicity=periodicity,
        )
        layer_list.append(input_layer)
        for i in range(hidden_layers):
            if normalization is not None:
                layer_list.append(normalization)
            if non_linearity is not None:
                layer_list.append(non_linearity())

            hidden_layer = high_order_fc_layers(
                layer_type=layer_type,
                n=n_hidden,
                in_features=hidden_width,
                out_features=hidden_width,
                segments=hidden_segments,
                rescale_output=rescale_output,
                scale=scale,
                periodicity=periodicity,
            )
            layer_list.append(hidden_layer)

        if non_linearity is not None:
            layer_list.append(non_linearity())
        if non_linearity is not None:
            layer_list.append(non_linearity())
        output_layer = high_order_fc_layers(
            layer_type=layer_type,
            n=n_out,
            in_features=hidden_width,
            out_features=out_width,
            segments=out_segments,
            rescale_output=rescale_output,
            scale=scale,
            periodicity=periodicity,
        )
        layer_list.append(output_layer)
        print("layer_list", layer_list)
        self.model = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class HighOrderFullyConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        layer_type: Union[List[str], str],
        n: List[int],
        channels: List[int],
        segments: List[int],
        kernel_size: List[int],
        rescale_output: bool = False,
        periodicity: float = None,
        normalization: Callable[[Any], Tensor] = None,
    ) -> None:
        """
        Args :

        """
        super().__init__()

        if len(channels) < 2:
            raise ValueError(
                f"Channels list must have at least 2 values [input_channels, output_channels]"
            )

        if (
            len(channels)
            == len(segments)
            == len(kernel_size)
            == len(layer_type)
            == len(n)
            is False
        ):
            raise ValueError(
                f"Lists for channels {len(channels)}, segments {len(segments)}, kernel_size {len(kernel_size)}, layer_type {len(layer_type)} and n {len(n)} must be the same size."
            )

        if len(channels) == len(n) + 1 is False:
            raise ValueError(
                f"Length of channels list {channels} should be one more than number of layers."
            )

        layer_list = []
        for i in range(len(channels) - 1):
            if normalization is not None:
                layer_list.append(normalization)

            layer = high_order_convolution_layers(
                layer_type=layer_type[i],
                n=n[i],
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_size[i],
                segments=segments[i],
                rescale_output=rescale_output,
                periodicity=periodicity,
            )
            layer_list.append(layer)

        self.model = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class HighOrderMLPMixerBlock(nn.Module):
    # Follow this block https://papers.nips.cc/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf
    pass


def interpolate_high_order_mlp(network_in: HighOrderMLP, network_out: HighOrderMLP):
    """
    Create a new network with weights interpolated from network_in.  If network_out has higher
    polynomial order than network_in then the output network will produce identical results to
    the input network, but be of higher polynomial order.  At this point the output network can
    be trained given the lower order network for weight initialization.  This technique is known
    as p-refinement (polynomial-refinement).

    Args :
        network_in : The starting network with some polynomial order n
        network_out : The output network.  This network should be initialized however its weights
        will be overwritten with interpolations from network_in
    """
    layers_in = [
        module
        for module in network_in.model.modules()
        if not isinstance(module, nn.Sequential)
    ]
    layers_out = [
        module
        for module in network_out.model.modules()
        if not isinstance(module, nn.Sequential)
    ]

    layer_pairs = zip(layers_in, layers_out)

    for l_in, l_out in layer_pairs:
        interpolate_polynomial_layer(l_in, l_out)
