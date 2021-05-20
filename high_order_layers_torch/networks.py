import torch.nn as nn
from torch import Tensor
from .layers import *
from typing import Any, Callable


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

        input_layer = high_order_fc_layers(layer_type=layer_type, n=n_in, in_features=in_width, out_features=hidden_width,
                                           segments=in_segments, rescale_output=rescale_output, scale=scale, periodicity=periodicity)
        layer_list.append(input_layer)
        for i in range(hidden_layers):
            if normalization is not None:
                layer_list.append(normalization)
            if non_linearity is not None:
                layer_list.append(non_linearity())

            hidden_layer = high_order_fc_layers(layer_type=layer_type, n=n_hidden, in_features=hidden_width, out_features=hidden_width,
                                                segments=hidden_segments, rescale_output=rescale_output, scale=scale, periodicity=periodicity)
            layer_list.append(hidden_layer)
        
        if non_linearity is not None:
                layer_list.append(non_linearity())
        if non_linearity is not None:
            layer_list.append(non_linearity())
        output_layer = high_order_fc_layers(layer_type=layer_type, n=n_out, in_features=hidden_width, out_features=out_width,
                                            segments=out_segments, rescale_output=rescale_output, scale=scale, periodicity=periodicity)
        layer_list.append(output_layer)
        print('layer_list', layer_list)
        self.model = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
