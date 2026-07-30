import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from torch.distributions import MultivariateNormal
from typing import Callable, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict, deque
from monai.networks.blocks import Convolution
from monai.networks.nets import VarAutoEncoder, AutoencoderKL
from torchvision.models import resnet18
from torchvision.transforms import v2
from geodesic_toolbox import *

from geodesic_toolbox import CoMetric
from monai.networks.blocks import Convolution
import torch.nn as nn

############################
# Base models
############################


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.mu_branch = ...
        self.log_var_branch = ...

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encodes a tensor x into the latent_code and the estimated log_variance.

        Parameters:
        ----------
        x : Tensor (batch_size, ...)
            The input tensor to be encoded.

        Returns:
        -------
        z : Tensor (batch_size, latent_dim)
            The latent code obtained from the encoder.
        log_variance : Tensor (batch_size, latent_dim)
        """

class BaseStochasticDecoder(nn.Module):
    def __init__(self):
        super(BaseStochasticDecoder, self).__init__()
        self.mu_branch = ...
        self.log_var_branch = ...

    def complete_forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decodes the latent code z into the reconstructed tensor and the estimated log_variance.

        Parameters:
        ----------
        z : Tensor (batch_size, latent_dim)
            The latent code to be decoded.

        Returns:
        -------
        x_reconstructed : Tensor (batch_size, ...)
            The reconstructed tensor obtained from the decoder.
        log_variance : Tensor (batch_size, ...)
            The estimated log variance of the reconstructed tensor.
        """
        ...

    def forward(self, z: Tensor) -> Tensor:
        """
        Decodes the latent code z into the reconstructed tensor.

        Parameters:
        ----------
        z : Tensor (batch_size, latent_dim)
            The latent code to be decoded.

        Returns:
        -------
        x_reconstructed : Tensor (batch_size, ...)
            The reconstructed tensor obtained from the decoder.
        """
        ...


class BaseDecoder(BaseStochasticDecoder):
    def __init__(self):
        super(BaseDecoder, self).__init__()
        self.log_var_branch = None

    def complete_forward(self, z):
        x_hat = self.forward(z)
        log_variance = torch.zeros_like(x_hat)
        return x_hat, log_variance


class BaseStochasticVAE(nn.Module):
    def __init__(self, encoder: BaseEncoder = None, decoder: BaseStochasticDecoder = None):
        super(BaseStochasticVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu: Tensor, log_variance: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, sigma^2) using N(0,1).

        Parameters:
        ----------
        mu : Tensor (batch_size, latent_dim)
            The mean of the latent code.
        log_variance : Tensor (batch_size, latent_dim)
            The log variance of the latent code.

        Returns:
        -------
        z : Tensor (batch_size, latent_dim)
            The sampled latent code.
        """
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return mu + eps * std

    def complete_forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the stochastic VAE model with all of the infos.

        Parameters:
        ----------
        x : Tensor (batch_size, ...)
            The input tensor to be encoded and decoded.

        Returns:
        -------
        x_reconstructed : Tensor (batch_size, ...)
            The reconstructed tensor obtained from the decoder.
        mu : Tensor (batch_size, latent_dim)
            The mean of the latent code obtained from the encoder.
        log_variance_latent : Tensor (batch_size, latent_dim)
            The estimated log variance of the latent code obtained from the encoder.
        log_variance_reconstructed : Tensor (batch_size, ...)
            The estimated log variance of the reconstructed tensor obtained from the decoder.
        """
        mu, log_variance_latent = self.encoder(x)
        if self.training:
            z = self.reparameterize(mu, log_variance_latent)
            x_reconstructed, log_variance_reconstructed = self.decoder.complete_forward(z)
            x_reconstructed = self.reparameterize(
                x_reconstructed, log_variance_reconstructed
            )
        else:
            z = mu
            x_reconstructed, log_variance_reconstructed = self.decoder.complete_forward(z)
        return x_reconstructed, mu, log_variance_latent, log_variance_reconstructed

    def forward(self,x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the stochastic VAE model that matches regular VAE forward signature.

        Parameters:
        ----------
        x : Tensor (batch_size, ...)
            The input tensor to be encoded and decoded.

        Returns:
        -------
        x_reconstructed : Tensor (batch_size, ...)
            The reconstructed tensor obtained from the decoder.
        mu : Tensor (batch_size, latent_dim)
            The mean of the latent code obtained from the encoder.
        log_variance_latent : Tensor (batch_size, latent_dim)
            The estimated log variance of the latent code obtained from the encoder.
        """
        x_reconstructed, mu, log_variance_latent, _ = self.complete_forward(x)
        return x_reconstructed, mu, log_variance_latent

class BaseVAE(BaseStochasticVAE):
    def __init__(self, encoder: BaseEncoder = None, decoder: BaseDecoder = None):
        super(BaseVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def complete_forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the stochastic VAE model with all of the infos.

        Parameters:
        ----------
        x : Tensor (batch_size, ...)
            The input tensor to be encoded and decoded.

        Returns:
        -------
        x_reconstructed : Tensor (batch_size, ...)
            The reconstructed tensor obtained from the decoder.
        mu : Tensor (batch_size, latent_dim)
            The mean of the latent code obtained from the encoder.
        log_variance_latent : Tensor (batch_size, latent_dim)
            The estimated log variance of the latent code obtained from the encoder.
        log_variance_reconstructed : Tensor (batch_size, ...)
            Used for compatibility with BaseStochasticVAE. Will be a zero tensor of the same shape as x_reconstructed.
            This follows from having as prior p(x|z) = N(x|decoder(z), I) in the VAE model.
        """
        mu, log_variance_latent = self.encoder(x)
        if self.training:
            z = self.reparameterize(mu, log_variance_latent)
        else:
            z = mu
        x_reconstructed, log_variance_reconstructed = self.decoder.complete_forward(z)
        return x_reconstructed, mu, log_variance_latent, log_variance_reconstructed

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the VAE model.

        Parameters:
        ----------
        x : Tensor (batch_size, ...)
            The input tensor to be encoded and decoded.

        Returns:
        -------
        x_reconstructed : Tensor (batch_size, ...)
            The reconstructed tensor obtained from the decoder.
        mu : Tensor (batch_size, latent_dim)
            The mean of the latent code obtained from the encoder.
        log_variance : Tensor (batch_size, latent_dim)
            The estimated log variance of the latent code.
        """
        mu, log_variance = self.encoder(x)
        if self.training:
            z = self.reparameterize(mu, log_variance)
        else:
            z = mu
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_variance





#############################
# MLP models
#############################


class Encoder_MLP(BaseEncoder):
    def __init__(
        self, input_dim, latent_dim: int, hidden_layers: tuple[int, ...] = (500, 500)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_dim_ = np.prod(input_dim) if isinstance(input_dim, tuple) else input_dim
        self.latent_dim = latent_dim

        self.layers = [torch.nn.Flatten()]
        for i, hidden_dim in enumerate(hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(self.input_dim_, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)

        self.mu = nn.Linear(hidden_layers[-1], self.latent_dim)
        self.std = nn.Linear(hidden_layers[-1], self.latent_dim)

        self.mu_branch = nn.Sequential(self.layers, self.mu)
        self.log_variance = nn.Sequential(self.layers, self.std)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.layers(x)
        mu = self.mu(z)
        log_variance = self.std(z)
        return mu, log_variance


class Decoder_MLP(BaseDecoder):
    def __init__(
        self, input_dim, latent_dim: int, hidden_layers: tuple[int, ...] = (500, 500)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_dim_flat = np.prod(input_dim) if isinstance(input_dim, tuple) else input_dim

        self.layers = []
        for i, hidden_dim in enumerate(hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(latent_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_layers[-1], self.input_dim_flat))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, z: Tensor) -> Tensor:
        x_reco = self.layers(z)
        x_reco = x_reco.reshape(-1, *self.input_dim)  # Reshape to original input dimensions
        return x_reco


class VAE_MLP(BaseVAE):
    def __init__(
        self,
        input_dim,
        latent_dim: int,
        hidden_layers: tuple[int, ...] = (128, 500),
    ):
        encoder = Encoder_MLP(
            input_dim=input_dim, latent_dim=latent_dim, hidden_layers=hidden_layers
        )
        decoder = Decoder_MLP(
            input_dim=input_dim, latent_dim=latent_dim, hidden_layers=hidden_layers
        )
        super().__init__(encoder=encoder, decoder=decoder)


class StochasticDecoder_MLP(BaseStochasticDecoder):
    def __init__(
        self, input_dim, latent_dim: int, hidden_layers: tuple[int, ...] = (500, 500)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_dim_flat = np.prod(input_dim) if isinstance(input_dim, tuple) else input_dim

        self.layers = []
        for i, hidden_dim in enumerate(hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(latent_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_dim))
            self.layers.append(nn.ReLU())
        self.mu = nn.Linear(hidden_layers[-1], self.input_dim_flat)
        self.std = nn.Linear(hidden_layers[-1], self.input_dim_flat)

        self.mu_branch = nn.Sequential(*self.layers, self.mu)
        self.log_variance = nn.Sequential(*self.layers, self.std)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        x_reco_mu = self.mu_branch(z)
        x_reco_log_variance = self.log_variance(z)
        return x_reco_mu, x_reco_log_variance


class StochasticVAE_MLP(BaseStochasticVAE):
    def __init__(
        self,
        input_dim,
        latent_dim: int,
        hidden_layers: tuple[int, ...] = (128, 500),
    ):
        encoder = Encoder_MLP(
            input_dim=input_dim, latent_dim=latent_dim, hidden_layers=hidden_layers
        )
        decoder = StochasticDecoder_MLP(
            input_dim=input_dim, latent_dim=latent_dim, hidden_layers=hidden_layers
        )
        super().__init__(encoder=encoder, decoder=decoder)


##############################
# Small convolutional models for MNIST and CIFAR10
##############################


class SmallConvBackbone(BaseEncoder):
    """
    Simple convolutional vae backbone model for MNIST and CIFAR10
    It expects to receive image of shape (B, C, 32, 32) where
    - B is the batch size
    - C is either 1 (MNIST) or 3 (CIFAR10)

    Params:
    -------
    latent_dim : int
        Dimension of the latent space
    n_channels : int
        Number of channels of the image (BW or RBG)

    Returns:
    -------
    mu_ : Tensor (B, latent_dim)
        The mean of the latent code
    log_variance_ : Tensor (B, latent_dim)
        The log variance of the latent code
    """

    def __init__(self, latent_dim: int, n_channels=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_channels = n_channels

        self.l1 = nn.Sequential(
            nn.Conv2d(self.n_channels, 128, kernel_size=(4, 4), stride=2),
            nn.InstanceNorm2d(num_features=128),
            nn.Softplus(),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2),
            nn.InstanceNorm2d(num_features=256),
            nn.Softplus(),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2),
            nn.InstanceNorm2d(num_features=512),
            nn.Softplus(),
        )
        self.mu = nn.Sequential(nn.Flatten(), nn.Linear(2048, self.latent_dim))
        self.log_variance = nn.Sequential(nn.Flatten(), nn.Linear(2048, self.latent_dim))

        self.mu_branch = nn.Sequential(self.l1, self.l2, self.l3, self.mu)
        self.log_std_branch = nn.Sequential(self.l1, self.l2, self.l3, self.log_variance)

    def forward(self, x: Tensor):
        z = self.l1(x)
        z = self.l2(z)
        z = self.l3(z)
        mu_ = self.mu(z)
        log_variance_ = self.log_variance(z)
        return mu_, log_variance_


class SmallConvDecoder(BaseDecoder):
    """
    Simple convolutional decoder model for MNIST and CIFAR10
    It expects to receive latent code of shape (B, latent_dim).

    Params:
    -------
    latent_dim : int
        Dimension of the latent space
    out_channels : int
        Number of channels of the image (BW or RBG)

    Returns:
    -------
    x_hat : Tensor (B, out_channels, 32, 32)
        The reconstructed image
    """

    def __init__(self, latent_dim: int, out_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.l1 = nn.Sequential(
            nn.Linear(self.latent_dim, 1024 * 8 * 8),
            nn.Unflatten(dim=1, unflattened_size=(1024, 8, 8)),
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=2),
            nn.InstanceNorm2d(num_features=512),
            nn.Softplus(),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=2),
            nn.InstanceNorm2d(num_features=256),
            nn.Softplus(),
        )
        self.mu = nn.Sequential(
            nn.ConvTranspose2d(256, self.out_channels, kernel_size=(1, 1), stride=1),
            nn.InstanceNorm2d(num_features=self.out_channels),
            nn.Sigmoid(),
        )
        self.decoder_branch = nn.Sequential(self.l1, self.l2, self.l3, self.mu)

    def forward(self, z: Tensor):
        x_hat = self.decoder_branch(z)
        return x_hat


class SmallConvVAE(BaseVAE):
    def __init__(self, latent_dim: int, n_channels: int = 1):
        encoder = SmallConvBackbone(latent_dim=latent_dim, n_channels=n_channels)
        decoder = SmallConvDecoder(latent_dim=latent_dim, out_channels=n_channels)
        super().__init__(encoder=encoder, decoder=decoder)


########################
# Conv VAE models for images of shape (256, 256)
########################
class ConvBlock(nn.Module):
    """Conv2d - InstanceNorm - PReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.PReLU(num_parameters=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvTransposeBlock(nn.Module):
    """ConvTranspose2d - InstanceNorm - PReLU"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.PReLU(num_parameters=1)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsample - Conv2d - InstanceNorm - PReLU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        upsample=True,
        affine: bool = False,
    ):
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="nearest") if upsample else nn.Identity()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=affine)
        self.activation = nn.PReLU(num_parameters=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvEncoder(nn.Module):
    """Encoder network that downsamples from 256x256 to 16x16"""

    def __init__(self, dropout: float = 0.0, in_channels: int = 1):
        super().__init__()
        # 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 16x16
        self.conv1 = ConvBlock(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.conv5(x)
        return x


class ConvDecoder(nn.Module):
    """Decoder network that upsamples from 16x16 to 256x256"""

    def __init__(self, dropout: float = 0.0, in_channels: int = 1, affine: bool = False):
        super().__init__()
        # 16x16 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.block1 = UpsampleBlock(256, 128, upsample=False, affine=affine)  # No upsample
        self.block2 = UpsampleBlock(128, 64, upsample=True, affine=affine)
        self.block3 = UpsampleBlock(64, 32, upsample=True, affine=affine)
        self.block4 = UpsampleBlock(32, 16, upsample=True, affine=affine)
        # self.block5 = UpsampleBlock(16, in_channels, upsample=True)

        self.upsample_final = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_final = nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = self.block4(x)
        x = self.dropout(x)
        x = self.upsample_final(x)
        x = self.conv_final(x)
        x = self.sigmoid(x)
        return x

class VAE(BaseVAE):
    def __init__(
        self, latent_dim: int, dropout: float = 0.0, in_channels: int = 1, affine: bool = False
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_ = ConvEncoder(dropout=dropout, in_channels=in_channels)
        self.decoder_ = ConvDecoder(dropout=dropout, in_channels=in_channels, affine=affine)

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)

        class EncoderWrapper(BaseEncoder):
            def __init__(self, encoder, mu, logvar):
                super().__init__()
                self.encoder = encoder
                self.mu = mu
                self.logvar = logvar

            def mu_branch(self, x):
                z = self.encoder(x)
                z = torch.flatten(z, start_dim=1)
                mu = self.mu(z)
                return mu

            def log_var_branch(self, x):
                z = self.encoder(x)
                z = torch.flatten(z, start_dim=1)
                logvar = self.logvar(z)
                return logvar

            def forward(self, x):
                z = self.encoder(x)
                z = torch.flatten(z, start_dim=1)
                mu = self.mu(z)
                logvar = self.logvar(z)
                return mu, logvar

        class DecoderWrapper(BaseDecoder):
            def __init__(self, decoder, fc_decode):
                super().__init__()
                self.decoder = decoder
                self.fc_decode = fc_decode

            def forward(self, z):
                z = self.fc_decode(z)
                z = torch.unflatten(z, dim=-1, sizes=(256, 16, 16))
                x_hat = self.decoder(z)
                return x_hat

        self.encoder = EncoderWrapper(self.encoder_, self.fc_mu, self.fc_logvar)
        self.decoder = DecoderWrapper(self.decoder_, self.fc_decode)


#############################
# 3D Conv VAE models
#############################


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class Encoder3D(BaseEncoder):
    """
    3D Convolutional Encoder for 3D VAE.

    Parameters:
    ----------
    latent_dim : int
        Dimension of the latent space.
    in_channels : int
        Number of input channels, default is 1 for grayscale images.
    hidden_channels : tuple[int, ...]
        Tuple specifying the number of channels in each convolutional block, default is (16, 32, 64, 128).
        Must be of length 4 to match the number of downsampling blocks.
    input_shape : tuple[int, int, int]
        Shape of the input image (H, W, D), default is (80, 80, 10).
    anisotropic : bool
        Whether to use anisotropic downsampling (default: True). If True, depth is downsampled less aggressively than height and width.
    use_resblocks : bool
        Whether to include residual blocks after each convolutional block (default: True).
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        hidden_channels: tuple[int, ...] = (16, 32, 64, 128),
        input_shape: tuple[int, int, int] = (80, 80, 10),
        anisotropic: bool = True,
        use_resblocks: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        if anisotropic:
            # Keeps depth less aggressively compressed
            # Hence the shapes after each block will be
            # (80,80,10)->(40,40,10)->(20,20,10)->(10,10,5)->(5,5,3)
            strides = ((2, 2, 1), (2, 2, 1), (2, 2, 2), (2, 2, 2))
        else:
            # Isotropic downsampling
            # The shapes after each block will be
            # (80,80,10)->(40,40,5)->(20,20,3)->(10,10,2)->(5,5,1)
            strides = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))

        assert len(hidden_channels) == len(
            strides
        ), "hidden_channels and stride schedule mismatch"

        blocks = []
        c_in = in_channels
        for c_out, s in zip(hidden_channels, strides):
            layers = [
                nn.Conv3d(c_in, c_out, kernel_size=3, stride=s, padding=1, bias=False),
                # nn.BatchNorm3d(c_out),
                nn.InstanceNorm3d(c_out, affine=True),
                nn.ReLU(inplace=True),
            ]
            if use_resblocks:
                layers.append(ResBlock3D(c_out, kernel_size=3))
            blocks.append(nn.Sequential(*layers))
            c_in = c_out

        self.features = nn.Sequential(*blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            feat = self.features(dummy)
            self.flat_dim = int(feat.numel())

        self.mu_branch = nn.Sequential(nn.Flatten(), nn.Linear(self.flat_dim, latent_dim))

        self.log_var_branch = nn.Sequential(nn.Flatten(), nn.Linear(self.flat_dim, latent_dim))
        # self.mu_branch = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.flat_dim, self.flat_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.flat_dim // 2, latent_dim),
        # )
        # self.log_var_branch = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.flat_dim, self.flat_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.flat_dim // 2, latent_dim),
        # )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.features(x)
        mu = self.mu_branch(h)
        logvar = self.log_var_branch(h)
        return mu, logvar


class Decoder3D(BaseDecoder):
    """
    3D Convolutional Decoder for 3D VAE.

    Parameters:
    ----------
    latent_dim : int
        Dimension of the latent space.
    out_channels : int
        Number of output channels, default is 1 for grayscale images.
    hidden_channels : tuple[int, ...]
        Tuple specifying the number of channels in each convolutional block, default is (128, 64, 32, 16).
        Must be of length 4 to match the number of upsampling blocks.
    anisotropic : bool
        Whether to use anisotropic upsampling (default: True). If True, depth is upsampled less aggressively than height and width.
    use_resblocks : bool
        Whether to include residual blocks after each convolutional block (default: True).
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int = 1,
        hidden_channels: tuple[int, ...] = (128, 64, 32, 16),
        anisotropic: bool = True,
        use_resblocks: bool = True,
        decoder_dropout: float = 0.1,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
        self.decoder_dropout = decoder_dropout
        if anisotropic:
            bottleneck_shape = (hidden_channels[0], 5, 5, 3)
        else:
            bottleneck_shape = (hidden_channels[0], 5, 5, 1)
        self.bottleneck_shape = bottleneck_shape
        c0, h0, w0, d0 = bottleneck_shape
        self.fc = nn.Linear(latent_dim, c0 * h0 * w0 * d0)

        if anisotropic:
            # Mirrors encoder schedule above with exact target sizes:
            # (5,5,3)->(10,10,5)->(20,20,10)->(40,40,10)->(80,80,10)
            upsample_sizes = ((10, 10, 5), (20, 20, 10), (40, 40, 10), (80, 80, 10))
        else:
            # Mirrors encoder schedule above with exact target sizes:
            # (5,5,1)->(10,10,2)->(20,20,3)->(40,40,5)->(80,80,10)
            upsample_sizes = ((10, 10, 2), (20, 20, 3), (40, 40, 5), (80, 80, 10))

        assert len(hidden_channels) == len(
            upsample_sizes
        ), "hidden_channels and upsample schedule mismatch"

        blocks = [nn.Unflatten(dim=-1, unflattened_size=bottleneck_shape)]
        c_in = c0
        for c_out, target_size in zip(hidden_channels, upsample_sizes):
            layers = [
                nn.Upsample(size=target_size, mode="trilinear", align_corners=False),
                nn.Conv3d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm3d(c_out),
                nn.InstanceNorm3d(c_out, affine=True),
                nn.ReLU(inplace=True),
            ]
            if use_resblocks:
                layers.append(ResBlock3D(c_out, kernel_size=3))
            blocks.append(nn.Sequential(*layers))
            c_in = c_out

        self.up_blocks = nn.Sequential(*blocks)

        self.last_block = nn.Sequential(
            nn.Conv3d(c_in, out_channels, kernel_size=3, stride=1, padding=1),
            output_activation if output_activation is not None else nn.Identity(),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z)
        h = self.up_blocks(h)
        if self.decoder_dropout > 0:
            h = F.dropout3d(h, p=self.decoder_dropout, training=self.training)
        x_hat = self.last_block(h)
        return x_hat


class VAE3D(BaseVAE):
    """
    3D Variational Autoencoder for volumetric data.

    Parameters:
    -----------
    latent_dim : int
        Dimension of the latent space (default: 16).
    in_channels : int
        Number of input channels (default: 1 for grayscale).
    out_channels : int
        Number of output channels (default: 1 for grayscale).
    encoder_channels : tuple[int, ...]
        Number of filters in encoder blocks (default: (16, 32, 64, 128)).
    decoder_channels : tuple[int, ...]
        Number of filters in decoder blocks (default: (128, 64, 32, 16)).
    anisotropic : bool
        Whether to use anisotropic convolution (default: True).
    use_resblocks : bool
        Whether to use residual blocks (default: True).
    output_activation : nn.Module | None
        Output activation function (default: None).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: tuple[int, ...] = (16, 32, 64, 128),
        decoder_channels: tuple[int, ...] = (128, 64, 32, 16),
        anisotropic: bool = True,
        use_resblocks: bool = True,
        decoder_dropout: float = 0.1,
        output_activation: nn.Module | None = None,
    ):
        """
        Parameters:
        -----------
        latent_dim : int
            Dimension of the latent space (default: 16).
        in_channels : int
            Number of input channels (default: 1 for grayscale).
        out_channels : int
            Number of output channels (default: 1 for grayscale).
        encoder_channels : tuple[int, ...]
            Number of filters in encoder blocks (default: (16, 32, 64, 128)).
        decoder_channels : tuple[int, ...]
            Number of filters in decoder blocks (default: (128, 64, 32)).
        anisotropic : bool
            Whether to use anisotropic convolution (default: True).
        use_resblocks : bool
            Whether to use residual blocks (default: True).
        output_activation : nn.Module | None
            Output activation function (default: None).
        """
        encoder = Encoder3D(
            latent_dim=latent_dim,
            in_channels=in_channels,
            hidden_channels=encoder_channels,
            anisotropic=anisotropic,
            use_resblocks=use_resblocks,
        )
        decoder = Decoder3D(
            latent_dim=latent_dim,
            out_channels=out_channels,
            hidden_channels=decoder_channels,
            anisotropic=anisotropic,
            use_resblocks=use_resblocks,
            decoder_dropout=decoder_dropout,
            output_activation=output_activation,
        )
        super().__init__(encoder, decoder)
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels


##############################
# MLP models for the Rander metrics
###############################


class HyperPlanPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        # base_cometric: CoMetric,
        hidden_dims: list[int] = [32, 64, 32],
    ):
        super(HyperPlanPredictor, self).__init__()
        self.latent_dim = latent_dim
        # self.base_cometric = base_cometric

        self.layers = nn.Sequential()
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.add_module(f"linear_{i}", nn.Linear(self.latent_dim, dim))
            else:
                self.layers.add_module(f"linear_{i}", nn.Linear(hidden_dims[i - 1], dim))
            self.layers.add_module(f"relu_{i}", nn.ReLU())
        self.layers.add_module(f"linear_out", nn.Linear(hidden_dims[-1], self.latent_dim))

    def forward(self, x):
        omega = self.layers(x)
        # Compute |w(x)|_{base_cometric^{-1}(x)}
        # if not self.train:
        #     # We do that because otherwise the gradient
        #     # Is messed up during learning.
        #     # The normalization is not needed at training time
        #     omega_norm = self.base_cometric.cometric(x, omega).sqrt()
        #     # Normalize
        #     omega = omega / (omega_norm[:, None] + 1e-8)
        return omega


class AccuracyPredictor(nn.Module):
    """
    Network to predict the accuracy of a model based on the latent code.

    Parameters:
    ----------
    latent_dim : int
        Dimension of the latent space.
    hidden_dims : list[int]
        List of hidden dimensions for the network. Default is [32, 128, 32].
    """

    def __init__(self, latent_dim: int, hidden_dims: list[int] = [32, 128, 32]):
        super(AccuracyPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential()
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.add_module(f"linear_{i}", nn.Linear(self.latent_dim, dim))
            else:
                self.layers.add_module(f"linear_{i}", nn.Linear(hidden_dims[i - 1], dim))
            self.layers.add_module(f"relu_{i}", nn.ReLU())
        self.layers.add_module(f"linear_out", nn.Linear(hidden_dims[-1], 1))
        self.layers.add_module("sigmoid", nn.Sigmoid())

    def forward(self, z):
        return self.layers(z)


class DummyAccuracyPredictor(nn.Module):
    """
    Dummy accuracy predictor that always returns 1.0.
    This is used as a placeholder when no accuracy prediction is needed.
    """

    def __init__(self):
        super(DummyAccuracyPredictor, self).__init__()

    def forward(self, z):
        return torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)


class Omega(torch.nn.Module):
    """
    Network to wrap the direction and amplitude of the Randers 1-form.
    The direction is normalized by the cometric so that the norm
    is only a function of the amplitude.

    Parameters:
    ----------
    direction : torch.nn.Module
        A neural network that predicts the direction of the Randers 1-form.
    amplitude : torch.nn.Module
        A neural network that predicts the amplitude of the Randers 1-form.
    cometric : CoMetric
        The cometric used to normalize the direction.
    """

    def __init__(
        self, direction: torch.nn.Module, amplitude: torch.nn.Module, cometric: CoMetric
    ):
        super().__init__()
        self.direction = direction
        self.amplitude = amplitude
        self.cometric = cometric

    def forward(self, X):
        rho = self.amplitude(X).clip(0.0, 1.0)  # Just in case
        omega = self.direction(X)
        omega_norm = self.cometric.cometric(X, omega).sqrt()
        omega = omega / (omega_norm[:, None] + 1e-8)
        return rho * omega


##############################
# Other models
################################


class ConvRegressor(nn.Module):
    """
    A simple convolutional regressor from the image space to R^d.
    It expects to receive images of shape (B, C, H, W)
    """

    def __init__(self, in_channels: int = 1, out_feature: int = 1):
        super(ConvRegressor, self).__init__()
        self.out_feature = out_feature
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.out_feature)
        self.resnet.conv1 = nn.Conv2d(
            in_channels,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias is not None,
        )
        self.resize = v2.Resize((224, 224))
        self.model = nn.Sequential(
            self.resize,
            self.resnet,
        )

        # Cast the weights to double precision
        for param in self.model.parameters():
            param.data = param.data.double()

    def forward(self, x: Tensor):
        # if self.out_feature == 1:
        #     return self.model(x).squeeze(-1)
        # else:
        return self.model(x)


class LatentClassifier(nn.Module):
    """
    A simple classifier for the latent space.
    It expects to receive latent code of shape (B, latent_dim).

    Parameters:
    ----------
    latent_dim : int
        Dimension of the latent space.
    num_classes : int
        Number of classes for classification.
    hidden_dim : list[int]
        List of hidden dimensions for the network. Default is [128, 64].
    Returns:
    -------
    logits : Tensor (B, num_classes)
        The logits for each class.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: list[int] = [128, 64],
        dropout: float = None,
    ):
        super(LatentClassifier, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if dropout is not None:
            assert dropout > 0 and dropout < 1, "Dropout must be between 0 and 1"

        self.fc = nn.Sequential()
        for i, dim in enumerate(hidden_dim):
            if i == 0:
                self.fc.add_module(f"linear_{i}", nn.Linear(self.latent_dim, dim))
            else:
                self.fc.add_module(f"linear_{i}", nn.Linear(hidden_dim[i - 1], dim))
            if self.dropout is not None:
                self.fc.add_module(f"dropout_{i}", nn.Dropout(p=self.dropout))

            self.fc.add_module(f"relu_{i}", nn.ReLU())
        self.fc.add_module(f"linear_out", nn.Linear(hidden_dim[-1], self.num_classes))

    def forward(self, z: Tensor):
        logits = self.fc(z)
        return logits


class LatentRegressor(LatentClassifier):
    """
    A simple regressor for the latent space.
    It expects to receive latent code of shape (B, latent_dim).

    Parameters:
    ----------
    latent_dim : int
        Dimension of the latent space.
    hidden_dim : list[int]
        List of hidden dimensions for the regressor.

    Returns:
    -------
    values : Tensor (B, 1)
        The predicted value for the regression task.
    """

    def __init__(self, latent_dim: int, hidden_dim: list[int] = [128, 64]):
        super().__init__(latent_dim=latent_dim, num_classes=1, hidden_dim=hidden_dim)
