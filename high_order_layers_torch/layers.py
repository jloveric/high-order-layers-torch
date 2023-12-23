import math

import torch.nn as nn
from torch.nn import Linear

from .FunctionalConvolution import *
from .FunctionalConvolutionTranspose import *
from .PolynomialLayers import *
from .ProductLayer import *
from .utils import (
    l2_normalization,
    max_abs_normalization_last,
    max_abs_normalization,
    max_abs_normalization_nd,
    max_center_normalization,
    max_center_normalization_last,
)


class MaxAbsNormalizationLast(nn.Module):
    """
    Normalize the last dimension of the input variable
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return max_abs_normalization_last(x, eps=self._eps)


class MaxCenterNormalizationLast(nn.Module):
    """
    Remove the average of the min and the max, center
    max abs normalization.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return max_center_normalization_last(x, eps=self._eps)


class MaxAbsNormalization(nn.Module):
    """
    Normalization for the 1D case (MLP)
    """

    def __init__(self, eps: float = 1e-6, dim: int = 1):
        super().__init__()
        self._eps = eps
        self._dim = dim

    def forward(self, x):
        return max_abs_normalization(x, eps=self._eps, dim=self._dim)


class MaxCenterNormalization(nn.Module):
    """
    Normalization for the 1D case (MLP) (x-avg)/(max(x)+eps) for each
    sample of the batch.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return max_center_normalization(x, eps=self._eps)


class MaxAbsNormalizationND(nn.Module):
    """
    Normalization for ND case, specifically convolutions
    but also works for MLP layers.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return max_abs_normalization_nd(x, eps=self._eps)


class L2Normalization(nn.Module):
    """
    L2 normalization for MLP layers
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return l2_normalization(x, eps=self._eps)


normalization_layers = {
    "max_abs": MaxAbsNormalization,
    "l2": L2Normalization,
}


def LinearAdapter(*args, in_features: int, out_features: int, **kwargs):
    return Linear(in_features=in_features, out_features=out_features, bias=True)


class LinearReluAdapter(nn.Module):
    def __init__(self, *args, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.f = Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x):
        return torch.nn.functional.relu(self.f.forward(x))


class SumLayer(nn.Module):
    def __init__(self, *args, layer_list: list[nn.Module], **kwargs):
        super().__init__()
        self._layer_list = layer_list

    def forward(self, x):
        x_all = [layer(x) for layer in self._layer_list]
        return torch.add(*x_all)


class SwitchLayer(Module):
    """
    Switch layer just creates 2 (or more) identical input layers
    and then multiplies the output of all those layers.  In effect
    one of the layers can turn of features of the other.
    """

    def __init__(
        self,
        layer_type: str,
        n: str,
        in_features: int,
        out_features: int,
        scale: float = 2.0,
        segments: int = None,
        normalization: Callable[[Any], Any] = None,
        num_input_layers: int = 2,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features

        self._layers = [
            high_order_fc_layers(
                layer_type=layer_type,
                n=n,
                in_features=in_features,
                out_features=out_features,
                segments=segments,
                rescale_output=False,
                scale=scale,
                periodicity=None,
                device=device,
            )
            for _ in range(num_input_layers)
        ]

        self._normalization = normalization

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features

    def forward(self, x) -> Tensor:
        outputs = [layer(x) for layer in self._layers]

        final = outputs[0]
        for i in range(1, len(outputs)):
            final *= outputs[i]

        if self._normalization is not None:
            final = self._normalization(final)

        return final

    def initialize(self, max_slope: int, max_offset: int):
        for layer in self._layers:
            initialize_polynomial_layer(
                layer_in=layer, max_slope=max_slope, max_offset=max_offset
            )

    def interpolate(
        self,
        layer_out: "SwitchLayer",
    ) -> None:
        for layer1, layer2 in zip(self._layers, layer_out._layers):
            layer1.interpolate(layer2)

    def refine(self, layer_out: "SwitchLayer"):
        for layer1, layer2 in zip(self._layers, layer_out._layers):
            layer1.refine(layer2)


def switch_continuous(**kwargs):
    return SwitchLayer(layer_type="continuous", **kwargs)


def switch_discontinuous(**kwargs):
    return SwitchLayer(layer_type="discontinuous", **kwargs)


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
    "switch_continuous": switch_continuous,
    "switch_discontinuous": switch_discontinuous,
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


def fixed_rotation_layer(n: int, rotations: int = 2, normalize: bool = True):
    """
    Take n inputs and compute all the variations, n_i+n_j, n_i-n_j
    and create a layer that computes these with fixed weights. For
    n=2, and rotations=2 outputs [x, t, a(x+t), a(x-t)].  Returns a fixed
    linear rotation layer (one that is not updated by gradients)
    Args :
        - n: The number of inputs, would be 2 for (x, t)
        - rotations: Number of rotations to apply pair by based on the inputs. So
        for input [x, y] and rotations=3, rotations are [x, y,a*(x+t), a*(x-t) ]
        - normalize: If true, normalizes values to be between -1 and 1
    Returns :
        A tuple containing the rotation layer and the output width of the layer
    """

    if rotations < 1:
        raise ValueError(
            f"Rotations must be 1 or greater. 1 represents no additional rotations. Got rotations={rotations}"
        )

    combos = []
    for i in range(n):
        reg = [0] * n
        reg[i] = 1.0
        combos.append(reg)

        for j in range(i + 1, n):
            for r in range(1, rotations):
                # We need to add rotations from each of 2 quadrants
                temp = [0] * n

                theta = (math.pi / 2) * (r / rotations)
                rot_x = math.cos(theta)
                rot_y = math.sin(theta)
                norm_val = 1 if normalize is False else abs(rot_x) + abs(rot_y)

                # Add the line and the line orthogonal
                temp[i] += rot_x / norm_val
                temp[j] += rot_y / norm_val

                combos.append(temp)

                other = [0] * n
                other[i] += rot_y / norm_val
                other[j] += -rot_x / norm_val

                combos.append(other)

    # 2 inputs, 1 rotation -> 2 combos
    # 2 inputs, 2 rotations -> 4 combos
    # 2 inputs, 3 rotations -> 6 combos
    # 2 inputs, 4 rotations -> 8 combos
    output_width = n + n * (n - 1) * (rotations - 1)
    layer = torch.nn.Linear(n, output_width, bias=False)
    weights = torch.tensor(combos)
    layer.weight = torch.nn.Parameter(weights, requires_grad=False)
    return layer, output_width


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
