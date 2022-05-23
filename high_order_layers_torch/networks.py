import torch.nn as nn
from torch import Tensor
from high_order_layers_torch.layers import (
    high_order_convolution_layers,
    high_order_convolution_transpose_layers,
    high_order_fc_layers,
)
from typing import Any, Callable, List, Union
from high_order_layers_torch.PolynomialLayers import interpolate_polynomial_layer
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pl_bolts.models.autoencoders.components import Interpolate

class Interpolate(nn.Module):
    """
    nn.Module wrapper for F.interpolate.
    copied from pl_bolts.models.autoencoders.components.Interpolate
    """

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)

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


def scalar_to_list(val : Union[List, str, int, float], size : int) :
    if val is None :
        return val

    if not isinstance(val, list) :
        return [val]*size
    return val


class HighOrderFullyConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        layer_type: Union[List[str], str],
        n: Union[List[int],int],
        channels: List[int],
        segments: Union[List[int], int],
        kernel_size: List[int],
        rescale_output: bool = False,
        periodicity: float = None,
        normalization: Callable[[Any], Tensor] = None,
        pooling : str = "2d",
        stride : List[int] = None,
    ) -> None:
        """
        Fully convolutional network is convolutions all the way down with global average pooling
        and flatten at the end.
        Args :
            layer_type : layer type [continuous2d, discontinous2d, ...]
            n : polynomial or fourier "order" [n1, n2, ...]
            channels : Number of channels for each layer (should be size layers+1) 
            segments : Number of segments in the polynomial if used at all as a list per layer.
            kernel_size : The kernel size for each layer
            rescale_output : whether to average the inputs to the next neuron
            periodicity : whether it should be periodic or not
            normalization : If not None, type of batch normalization to use.
            pooling : 1d, 2d or 3d (for the final average pool)
            stride : Stride for each layer
        """
        super().__init__()

        self.layer_type = layer_type
        self.n = n
        self.channels = channels
        self.segments = segments
        self.kernel_size = kernel_size
        self.stride = stride
        
        if len(channels) < 2:
            raise ValueError(
                f"Channels list must have at least 2 values [input_channels, output_channels]"
            )

        size = len(kernel_size)
        self.layer_type = scalar_to_list(self.layer_type, size)
        self.n = scalar_to_list(self.n, size)
        self.stride = scalar_to_list(self.stride, size)
        self.segments = scalar_to_list(self.segments, size)

        if (
            len(self.segments)
            == len(self.kernel_size)
            == len(self.layer_type)
            == len(self.n)
            is False
        ):
            raise ValueError(
                f"Lists for segments {len(self.segments)}, kernel_size {len(self.kernel_size)}, layer_type {len(self.layer_type)} and n {len(self.n)} must be the same size."
            )

        if len(self.channels) == len(self.n) + 1 is False:
            raise ValueError(
                f"Length of channels list {self.channels} should be one more than number of layers."
            )

        layer_list = []
        for i in range(len(self.channels) - 1):
            if normalization is not None:
                layer_list.append(normalization(self.channels[i]))

            layer = high_order_convolution_layers(
                layer_type=self.layer_type[i],
                n=self.n[i],
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=self.kernel_size[i],
                segments=self.segments[i],
                rescale_output=rescale_output,
                periodicity=periodicity,
                stride = 1 if self.stride is None else self.stride[i]
            )
            layer_list.append(layer)

        # Add an average pooling layer
        if pooling == "1d" :
            avg_pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == "2d" :
            avg_pool = nn.AdaptiveAvgPool2d((1,1))
        elif pooling =="3d" :
            avg_pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.model = nn.Sequential(*layer_list, avg_pool, nn.Flatten())

    def forward(self, x: Tensor) -> Tensor:
        temp = self.model(x)
        return temp

    @property
    def output_size(self) :
        return self.channels[-1] # avg pooling and flatten should give channels size


class HighOrderFullyDeconvolutionalNetwork(nn.Module):
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
        stride : List[int] = None,
    ) -> None:
        """
        Args :

        """
        super().__init__()

        self._layer_type = layer_type
        self._n = n
        self._channels = channels
        self._segments = segments
        self._kernel_size = kernel_size
        self._stride = stride
        
        if len(self._channels) < 2:
            raise ValueError(
                f"Channels list must have at least 2 values [input_channels, output_channels]"
            )

        size = len(self._channels)
        self._layer_type = scalar_to_list(self._layer_type, size)
        self._n = scalar_to_list(self._n, size)
        self._stride = scalar_to_list(self._stride, size)
        self._segments = scalar_to_list(self._segments, size)

        if (
           len(self._segments)
            == len(self._kernel_size)
            == len(self._layer_type)
            == len(self._n)
            is False
        ):
            raise ValueError(
                f"Lists for segments {len(self._segments)}, kernel_size {len(self._kernel_size)}, layer_type {len(self._layer_type)} and n {len(self._n)} must be the same size."
            )

        if len(self._channels) == len(self._n) + 1 is False:
            raise ValueError(
                f"Length of channels list {self._channels} should be one more than number of layers."
            )

        layer_list = []

        # Input is a flat vector
        self._in_channels = self._channels[0]

        for i in range(len(self._channels) - 1):
            if normalization is not None:
                #norm = nn.BatchNorm2d(self._channels[i])
                layer_list.append(normalization(self._channels[i]))

            layer = high_order_convolution_transpose_layers(
                layer_type=self._layer_type[i],
                n=self._n[i],
                in_channels=self._channels[i],
                out_channels=self._channels[i + 1],
                kernel_size=self._kernel_size[i],
                segments=self._segments[i],
                rescale_output=rescale_output,
                periodicity=periodicity,
                stride=1 if self._stride is None else self._stride[i]
            )
            layer_list.append(layer)

        self.model = nn.Sequential(*layer_list)

    @property
    def in_channels(self) :
        return self._in_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


# Copied from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
# and modified to work with arbitrary encoder / decoder so it that it works with
# high order networks.
class VanillaVAE(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        **kwargs,
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = encoder
        
        print('self.encoder.modules',self.encoder.modules)
        encoder_out_features = self.encoder.output_size

        print("encoder_out_features", encoder_out_features)
        self.fc_mu = nn.Linear(encoder_out_features, latent_dim)
        self.fc_var = nn.Linear(encoder_out_features, latent_dim)
        
        self.decoder = decoder
        self._in_features = self.decoder.in_channels # It's flat so channels is features
        self.decoder_input = nn.Linear(latent_dim, self._in_features)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        Args :
            input: (Tensor) Input tensor to encoder [N x C x H x W]
        Returns :
            (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        Args :
            z: (Tensor) [B x D]

        returns (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.decoder.in_channels, 1, 1)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        Args :
            mu: (Tensor) Mean of the latent Gaussian [B x D]
            logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        return (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        Args :
            args:
            kwargs:
        Returns:
        """

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        Args :
            num_samples: (Int) Number of samples
            current_device: (Int) Device to run the model
        Returns (Tensor)
        """
        print('latent_dim', self.latent_dim, self.device)
        z = torch.randn(num_samples, self.latent_dim, device=self.device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        Args :
            x: (Tensor) [B x C x H x W]
        Returns :
            (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


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
