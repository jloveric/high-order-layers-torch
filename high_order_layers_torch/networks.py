import torch
import torch.nn as nn
from torch import Tensor
from .layers import *


class HighOrderMLP(nn.Module):
    def __init__(
        self,
        layer_type: str,
        n: str,
        in_width: int,
        in_segments: int,
        out_segments: int,
        out_width: int,
        hidden_segments: int,
        hidden_layers: int,
        hidden_width: int,
        scale: float = 2.0,
        rescale_output: bool = False,
        periodicity: float = None,
        non_linearity=None
    ) -> None:
        super().__init__()
        layer_list = []
        input_layer = high_order_fc_layers(layer_type=layer_type, n=n, in_features=in_width, out_features=hidden_width,
                                           segments=in_segments, rescale_output=rescale_output, scale=scale, periodicity=periodicity)
        layer_list.append(input_layer)
        for i in range(hidden_layers):
            if non_linearity is not None:
                layer_list.append(non_linearity())
            hidden_layer = high_order_fc_layers(layer_type=layer_type, n=n, in_features=hidden_width, out_features=hidden_width,
                                                segments=hidden_segments, rescale_output=rescale_output, scale=scale, periodicity=periodicity)
            layer_list.append(hidden_layer)

        if non_linearity is not None:
            layer_list.append(non_linearity())
        output_layer = high_order_fc_layers(layer_type=layer_type, n=n, in_features=hidden_width, out_features=out_width,
                                            segments=out_segments, rescale_output=rescale_output, scale=scale, periodicity=periodicity)
        layer_list.append(output_layer)
        self.model = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
