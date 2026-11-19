import torch
import torch.nn.functional as F
from geodesic_toolbox import *
from geodesic_toolbox import CoMetric
from torch import Tensor, nn
from torchvision.models import resnet18
from torchvision.transforms import v2


def cdist(x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """
    Computes the pairwise distance between two sets of points.
    This implementation is a workaround for the fact that torch.cdist does not support forward AD.
    dst_mat[i, j] = ||x1[i] - x2[j]||_p = sqrt( ||x1[i]||_p^2 + ||x2[j]||_p^2 - 2 * <x1[i], x2[j]> )

    Parameters:
    ----------
    x1 : torch.Tensor (N, d)
        The first set of points.
    x2 : torch.Tensor (M, d)
        The second set of points.
    p : float
        The norm degree. Default is 2 (Euclidean distance).

    Returns:
    -------
    dst_mat : torch.Tensor (N, M)
        The pairwise distance matrix.
    """
    # diff = x1.unsqueeze(1) - x2.unsqueeze(0) # (N, M, d)
    # dst_mat = torch.norm(diff, p=p, dim=-1)
    x1_norm = torch.norm(x1, p=p, dim=-1, keepdim=True) ** 2  # (N, 1)
    x2_norm = torch.norm(x2, p=p, dim=-1, keepdim=True) ** 2  # (M, 1)
    inner_product = torch.matmul(x1, x2.T)  # (N, M)
    dst_mat = (x1_norm + x2_norm.T - 2 * inner_product).clamp_min(0.0).sqrt()  # (N, M)
    return dst_mat


############################
# Base models
############################


class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_branch: torch.nn.Module = ...
        self.log_var_branch: torch.nn.Module = ...

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
            The estimated log variance of the latent code.
        """


class LogVarToStd(nn.Module):
    def __init__(self, log_var_module: nn.Module):
        super().__init__()
        self.log_var_module = log_var_module

    def forward(self, x: Tensor) -> Tensor:
        log_var = self.log_var_module(x)
        std = torch.exp(0.5 * log_var)
        return std


class BaseStochasticDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_branch: torch.nn.Module = ...
        self.log_var_branch: torch.nn.Module = ...

        # This doesn't actually work you need to do that after the model is created because the log_var_branch is not defined yet
        # But you get the jist
        self.std_branch: torch.nn.Module = LogVarToStd(self.log_var_branch)

    def complete_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
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


class BaseDecoder(BaseStochasticDecoder):
    def __init__(self):
        super().__init__()
        self.log_var_branch = None

    def complete_forward(self, z):
        x_hat = self.forward(z)
        log_variance = torch.zeros_like(x_hat)
        return x_hat, log_variance


class BaseStochasticVAE(nn.Module):
    def __init__(self, encoder: BaseEncoder = None, decoder: BaseStochasticDecoder = None):
        super().__init__()
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

    def complete_forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
            x_reconstructed = self.reparameterize(x_reconstructed, log_variance_reconstructed)
        else:
            z = mu
            x_reconstructed, log_variance_reconstructed = self.decoder.complete_forward(z)
        return x_reconstructed, mu, log_variance_latent, log_variance_reconstructed

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def complete_forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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


class BaseSphericalVAE(BaseVAE):
    def __init__(self, encoder: BaseEncoder = None, decoder: BaseDecoder = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def mobius_transform(self, mu: Tensor, rho: Tensor, x: Tensor) -> Tensor:
        """
        Computes the Mobius transform of a point x on the sphere with respect to a mean mu and concentration rho.
        """
        num = x + rho * mu
        denom = 1 + 2 * rho * torch.sum(mu * x, dim=-1, keepdim=True) + rho**2
        y = (1 - rho**2) * num / denom + rho * mu
        return y

    def reparameterize(self, mu: Tensor, rho: Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from a spherical Cauchy distribution
        using the Mobius transform reparameterization.

        Parameters:
        ----------
        mu : Tensor (batch_size, latent_dim)
            The mean of the latent code.
            Must be on the unit sphere
        rho : Tensor (batch_size, 1)
            The concentration parameter of the spherical Cauchy distribution.
            Must be in the range (0, 1) for valid sampling.

        Returns:
        -------
        z : Tensor (batch_size, latent_dim)
            The sampled latent code on the sphere.
        """
        x = torch.randn_like(mu)  # Sample from standard normal distribution
        x = x / torch.norm(x, dim=-1, keepdim=True)  # Normalize to unit length
        z = self.mobius_transform(mu, rho, x)  # Apply Mobius transform
        return z

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        rho : Tensor (batch_size, 1)
            The estimated concentration parameter of the spherical Cauchy distribution.
        """
        mu, rho = self.encoder(x)
        if self.training:
            z = self.reparameterize(mu, rho)
        else:
            z = mu
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, rho


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
            self.layers.append(nn.SiLU())
        self.layers = nn.Sequential(*self.layers)

        self.mu = nn.Linear(hidden_layers[-1], self.latent_dim)
        self.std = nn.Linear(hidden_layers[-1], self.latent_dim)

        self.mu_branch = nn.Sequential(self.layers, self.mu)
        self.log_variance = nn.Sequential(self.layers, self.std)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.layers(x)
        mu = self.mu(z)
        log_variance = self.std(z)
        return mu, log_variance


class SphericalEncoder_MLP(BaseEncoder):
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
            self.layers.append(nn.SiLU())
        self.layers = nn.Sequential(*self.layers)

        self.mu = nn.Linear(hidden_layers[-1], self.latent_dim)
        self.rho = nn.Linear(hidden_layers[-1], 1)

        self.mu_branch = nn.Sequential(self.layers, self.mu)
        self.log_variance = nn.Sequential(self.layers, self.rho)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.layers(x)
        mu = self.mu(z)
        mu = nn.functional.normalize(mu, p=2, dim=1)
        rho = self.rho(z).sigmoid()  # Ensure rho is in (0, 1)
        return mu, rho


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
            self.layers.append(nn.SiLU())
        self.layers.append(nn.Linear(hidden_layers[-1], self.input_dim_flat))
        self.mu_branch = nn.Sequential(*self.layers)

    def forward(self, z: Tensor) -> Tensor:
        x_reco = self.mu_branch(z)
        x_reco = x_reco.reshape(-1, *self.input_dim)  # Reshape to original input dimensions
        return x_reco


class VAE_MLP(BaseVAE):
    def __init__(
        self, input_dim, latent_dim: int, hidden_layers: tuple[int, ...] = (128, 500)
    ):
        encoder = Encoder_MLP(
            input_dim=input_dim, latent_dim=latent_dim, hidden_layers=hidden_layers
        )
        decoder = Decoder_MLP(
            input_dim=input_dim, latent_dim=latent_dim, hidden_layers=hidden_layers[::-1]
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
            self.layers.append(nn.SiLU())
        self.mu = nn.Linear(hidden_layers[-1], self.input_dim_flat)
        self.std = nn.Linear(hidden_layers[-1], self.input_dim_flat)

        self.mu_branch = nn.Sequential(*self.layers, self.mu)
        self.log_variance = nn.Sequential(*self.layers, self.std)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        x = self.layers(z)
        x_reco_mu = self.mu(x)
        x_reco_log_variance = self.std(x)
        return x_reco_mu, x_reco_log_variance


class StochasticVAE_MLP(BaseStochasticVAE):
    def __init__(
        self, input_dim, latent_dim: int, hidden_layers: tuple[int, ...] = (128, 500)
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
        self.mu_branch = nn.Sequential(self.l1, self.l2, self.l3, self.mu)

    def forward(self, z: Tensor):
        x_hat = self.mu_branch(z)
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
    """Conv2d - InstanceNorm - SiLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvTransposeBlock(nn.Module):
    """ConvTranspose2d - InstanceNorm - SiLU"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsample - Conv2d - InstanceNorm - SiLU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        upsample_scale: float | None = None,
        affine: bool = False,
    ):
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=upsample_scale, mode="nearest")
            if upsample_scale
            else nn.Identity()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=affine)
        self.activation = nn.SiLU()

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
        self.block1 = UpsampleBlock(
            256, 128, upsample_scale=None, affine=affine
        )  # No upsample
        self.block2 = UpsampleBlock(128, 64, upsample_scale=2.0, affine=affine)
        self.block3 = UpsampleBlock(64, 32, upsample_scale=2.0, affine=affine)
        self.block4 = UpsampleBlock(32, 16, upsample_scale=2.0, affine=affine)
        # self.block5 = UpsampleBlock(16, in_channels, upsample_scale=2.0, affine=affine)

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
        self, latent_dim: int, dropout: float = 0.0, in_channels: int = 1, affine: bool = True
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
                self.unflatten = nn.Unflatten(dim=-1, unflattened_size=(256, 16, 16))
                self.fc_decode = fc_decode
                self.mu_branch = nn.Sequential(self.fc_decode, self.unflatten, self.decoder)

            def forward(self, z):
                return self.mu_branch(z)

        self.encoder = EncoderWrapper(self.encoder_, self.fc_mu, self.fc_logvar)
        self.decoder = DecoderWrapper(self.decoder_, self.fc_decode)


class SphericalVAE(BaseSphericalVAE):
    def __init__(
        self, latent_dim: int, dropout: float = 0.0, in_channels: int = 1, affine: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_ = ConvEncoder(dropout=dropout, in_channels=in_channels)
        self.decoder_ = ConvDecoder(dropout=dropout, in_channels=in_channels, affine=affine)

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_rho = nn.Linear(256 * 16 * 16, 1)
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)

        class EncoderWrapper(BaseEncoder):
            def __init__(self, encoder, mu, rho):
                super().__init__()
                self.encoder = encoder
                self.mu = mu
                self.logvar = rho

            def mu_branch(self, x):
                z = self.encoder(x)
                z = torch.flatten(z, start_dim=1)
                mu = self.mu(z)
                mu = torch.nn.functional.normalize(mu, p=2, dim=1)
                return mu

            def log_var_branch(self, x):
                z = self.encoder(x)
                z = torch.flatten(z, start_dim=1)
                logvar = self.logvar(z).sigmoid()  # Ensure rho is in (0, 1)
                return logvar

            def forward(self, x):
                z = self.encoder(x)
                z = torch.flatten(z, start_dim=1)
                mu = self.mu(z)
                mu = torch.nn.functional.normalize(mu, p=2, dim=1)
                rho = self.logvar(z).sigmoid()  # Ensure rho is in (0, 1)
                return mu, rho

        class DecoderWrapper(BaseDecoder):
            def __init__(self, decoder, fc_decode):
                super().__init__()
                self.decoder = decoder
                self.unflatten = nn.Unflatten(dim=-1, unflattened_size=(256, 16, 16))
                self.fc_decode = fc_decode
                self.mu_branch = nn.Sequential(self.fc_decode, self.unflatten, self.decoder)

            def forward(self, z):
                return self.mu_branch(z)

        self.encoder = EncoderWrapper(self.encoder_, self.fc_mu, self.fc_rho)
        self.decoder = DecoderWrapper(self.decoder_, self.fc_decode)


class PositiveLinear(nn.Module):
    """
    Positive Linear layer that ensures the weights and biases are positive by applying a softplus activation to them.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.b = nn.Parameter(torch.randn(out_features))
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        W_pos = F.softplus(self.W)
        b_pos = F.softplus(self.b) if self.b is not None else None
        return F.linear(x, W_pos, b_pos)


class RBFLayer(nn.Module):
    """
    RBF Layer that computes:
    v_k(x) = exp(-||x - c_k||^2 / (2 * tau_k^2)) for k = 1, ..., K
    where c_k are the centroids and tau_k are the bandwidths.

    Parameters:
    ----------
    data : Tensor (N, d)
        The data points used to compute the centroids and bandwidths.
    K : int
        The number of centroids to use. If K=-1, use all data points as centroids.
    kappa : float
        The scaling factor for the bandwidths. The bandwidths are computed as tau_k^2
    """

    def __init__(self, data: torch.Tensor, K: int, kappa: float = 1.0):
        super().__init__()
        self.in_features = data.shape[1]

        self.register_buffer("kappa", Tensor([kappa]))
        self.register_buffer("centroids", data)
        self.register_buffer(
            "tau", torch.ones(data.shape[0], device=data.device, dtype=data.dtype)
        )

        if K is not None:
            if K > data.shape[0]:
                print(
                    f"Warning: K={K} is greater than the number of data points {data.shape[0]}. Using all data points."
                )
                K = data.shape[0]
            self.process_centroids(K)

        self.K = self.centroids.shape[0]

        tau_squared = self.compute_bandwidths(data, kappa)
        self.register_buffer("tau_squared", tau_squared)

    def process_centroids(self, K: int) -> None:
        """
        Process the centroids to select K representative centroids using K-Medoids clustering.

        Parameters:
        K : int
            The number of centroids to select. If K=-1, use all centroids.
        """
        # dst_mat = torch.cdist(self.centroids, self.centroids, p=2).cpu().numpy()
        dst_mat = cdist(self.centroids, self.centroids, p=2).cpu().numpy()
        kmedoids_model = kmedoids.KMedoids(
            n_clusters=K, metric="precomputed", random_state=1312
        )
        kmedoids_model.fit(dst_mat)
        self.centroids = self.centroids[kmedoids_model.medoid_indices_]

    def compute_bandwidths(
        self,
        data: Tensor,
        kappa: Tensor,
        min_cluster_size: int = 3,
        neighbor_rank: int = 5,
        min_scale_quantile: float = 0.25,
        max_scale_quantile: float = 0.95,
        eps: float = 1e-12,
    ) -> Tensor:
        """
        Compute the (squared) bandwidths of the RBF kernels as:

            tau_k^2 = kappa * scale_k^2

        where scale_k^2 is an adaptive local-scale estimate for centroid c_k,
        blended between two sources:

        1. Local in-cluster scale (reliable when the cluster has enough points):
            local_scale_k^2 = (1 / Card(C_k)) * sum_{x_i in C_k} ||c_k - x_i||^2
        where C_k is the set of data points assigned to centroid c_k (i.e. c_k
        is their nearest centroid).

        2. Neighbor-based robust scale (used as a fallback when Card(C_k) is
        small and the local estimate would be unreliable/pathological):
            nn_scale_k^2 = ||c_k - c_(neighbor_rank)||^2
        i.e. the squared distance from c_k to its `neighbor_rank`-th nearest
        neighboring centroid (not the 1st, to avoid overly peaked kernels).
        This is clamped between global quantiles of nn_scale^2 across all
        centroids (min_scale_quantile, max_scale_quantile) so that outlier
        centroids do not produce pathologically small or large bandwidths.

        The two estimates are blended per-centroid via a reliability weight
        r_k in [0, 1], based on the cluster cardinality relative to
        min_cluster_size:

            r_k     = clamp((Card(C_k) - 1) / (min_cluster_size - 1), 0, 1)
            scale_k^2 = r_k * local_scale_k^2 + (1 - r_k) * nn_scale_k^2

        The blended scale is then re-clamped to the same global [min, max]
        bounds, and the final bandwidth is:

            tau_k^2 = kappa * scale_k^2

        Parameters:
        -----------
        data : Tensor (N, d)
            The original data points, used to compute the clusters of centroids.
        kappa : Tensor
            The scaling factor applied to the blended scale to obtain tau^2.
        min_cluster_size : int
            Cluster cardinality at or above which the local in-cluster scale is
            trusted fully (reliability = 1).
        neighbor_rank : int
            Which nearest-neighbor centroid (1-indexed, excluding self) to use
            for the robust fallback scale.
        min_scale_quantile, max_scale_quantile : float
            Quantiles (over all centroids) used to clamp the neighbor-based
            scale, preventing outlier centroids from dominating.
        eps : float
            Small constant to avoid division by zero / degenerate scales.

        Returns:
        --------
        tau_squared : Tensor (K,)
            The computed squared bandwidths for each centroid.
        """
        K = self.centroids.shape[0]

        # Assign each sample to the closest centroid.
        # dst_data = torch.cdist(self.centroids, data, p=2)  # (K,N)
        dst_data = cdist(self.centroids, data, p=2)  # (K,N)
        closest_centroid = dst_data.argmin(dim=0)  # (N,)

        # Robust fallback scale from centroid geometry.
        # Use a higher-order neighbor to avoid over-peaked kernels when K is large.
        # dst_centroids = torch.cdist(self.centroids, self.centroids, p=2)
        dst_centroids = cdist(self.centroids, self.centroids, p=2)
        dst_centroids.fill_diagonal_(float("inf"))
        sorted_dst, _ = torch.sort(dst_centroids, dim=1)
        rank = min(max(neighbor_rank - 1, 0), max(K - 2, 0))
        nn_dist = sorted_dst[:, rank]  # (K,)
        nn_scale2 = nn_dist.pow(2).clamp_min(eps)

        # Global robust floor/ceiling so outlier clusters do not dominate smoothness.
        min_scale2 = torch.quantile(nn_scale2, min_scale_quantile).clamp_min(eps)
        max_scale2 = torch.quantile(nn_scale2, max_scale_quantile).clamp_min(eps)
        nn_scale2 = nn_scale2.clamp(min=min_scale2, max=max_scale2)

        # Local in-cluster scale. May be unreliable when cluster cardinality is very small.
        local_scale2 = torch.zeros(K, device=self.centroids.device, dtype=self.centroids.dtype)
        counts = torch.bincount(closest_centroid, minlength=K).to(self.centroids.dtype)
        for k in range(K):
            cluster_points = data[closest_centroid == k]  # (Card(C_k),d)
            if cluster_points.shape[0] > 0:
                c_k = self.centroids[k : k + 1]
                # dist2 = torch.cdist(c_k, cluster_points, p=2).pow(2)
                dist2 = cdist(c_k, cluster_points, p=2).pow(2)
                local_scale2[k] = dist2.mean().clamp_min(eps)

        # Blend local scale with centroid-neighborhood scale.
        # For small clusters (K ~ N), this prevents pathological very narrow kernels.
        reliability = ((counts - 1) / max(min_cluster_size - 1, 1)).clamp(0.0, 1.0)
        scale2 = reliability * local_scale2 + (1.0 - reliability) * nn_scale2
        scale2 = scale2.clamp(min=min_scale2, max=max_scale2)

        tau_squared = kappa * scale2

        # Fix any potential numerical issues with the bandwidths
        if not torch.isfinite(tau_squared).all():
            finite_mask = torch.isfinite(tau_squared)
            if finite_mask.any():
                fill_value = tau_squared[finite_mask].median()
            else:
                fill_value = torch.tensor(
                    1.0, device=tau_squared.device, dtype=tau_squared.dtype
                )
            tau_squared = torch.where(finite_mask, tau_squared, fill_value)
        return tau_squared

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the RBF layer.

        Parameters:
        ----------
        x : Tensor (batch_size, in_features)
            The input tensor to be transformed by the RBF layer.

        Returns:
        -------
        rbf_output : Tensor (batch_size, K)
            The output tensor after applying the RBF transformation.
        """
        # dists_squared = torch.cdist(x, self.centroids, p=2).pow(2)  # (batch_size, K)
        dists_squared = cdist(x, self.centroids, p=2).pow(2)  # (batch_size, K)
        rbf_output = torch.exp(-dists_squared / (2 * self.tau_squared))
        return rbf_output


class LinearVarianceDecoderBranch(nn.Module):
    """
    This layer compute the log variance through beta(z) = W v(z) + b, where v(z) is the output of the RBF layer and W, b are learnable positive parameters.
    And we have that beta(z) = 1/var(z) = 1/sigma^2(z), so that we can compute the log variance as log(sigma^2(z)) = -log(beta(z)) = -log(W v(z) + b)

    Parameters:
    ----------
    latent_dim : int
        Dimension of the latent space.
    output_shape : tuple[int, ...]
        Shape of the output tensor. Eg (C,H,W) for images or (D,) for 1D data.
    data : Tensor (N, latent_dim)
        The data points used to compute the centroids and bandwidths for the RBF layer.
    K : int
        The number of centroids to use in the RBF layer. If K=-1, use all data points as centroids.
    kappa : float
        The scaling factor for the bandwidths in the RBF layer.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, ...],
        data: Tensor,
        K: int,
        kappa: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.output_shape = output_shape
        self.flat_dim = np.prod(output_shape)

        self.v = RBFLayer(data=data, K=K, kappa=kappa)
        self.W = PositiveLinear(in_features=K, out_features=self.flat_dim)

    def forward(self, z: Tensor) -> Tensor:
        v_z = self.v(z)
        beta_z = self.W(v_z)
        log_variance_hat = -torch.log(beta_z + 1e-8)  # Add a small constant to avoid log(0)
        log_variance_hat = log_variance_hat.view(-1, *self.output_shape)
        return log_variance_hat


class VarianceDecoderBranch(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        data: Tensor,
        K: int,
        kappa: float = 1.0,
        affine: bool = True,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.linear_rbf = LinearVarianceDecoderBranch(
            latent_dim=latent_dim, output_shape=64 * 8 * 8, data=data, K=K, kappa=kappa
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 8, 8))
        self.block1 = UpsampleBlock(64, 32, upsample_scale=4.0, affine=affine)
        self.block2 = UpsampleBlock(32, 16, upsample_scale=4.0, affine=affine)
        self.block3 = UpsampleBlock(16, 8, upsample_scale=2.0, affine=affine)
        self.conv_final = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        if p_dropout > 0.0:
            self.dropout = nn.Dropout2d(p=p_dropout)
        else:
            self.dropout = nn.Identity()

        self.log_var_branch = nn.Sequential(
            self.linear_rbf,
            self.unflatten,
            self.block1,
            self.dropout,
            self.block2,
            self.dropout,
            self.block3,
            self.dropout,
            self.conv_final,
        )

    def forward(self, z: Tensor) -> Tensor:
        log_variance_hat = self.log_var_branch(z)
        return log_variance_hat


class StochasticVAE(BaseStochasticVAE):
    def __init__(self, base_model: VAE, data: Tensor, K: int, kappa: float = 1.0):
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.decoder.mu_branch = base_model.decoder.mu_branch

        # # Terrible idea it uses so much memory and parameters
        # self.decoder.log_var_branch = LinearVarianceDecoderBranch(
        #     latent_dim=base_model.latent_dim,
        #     output_shape=(1, 256, 256),
        #     data=data,
        #     K=K,
        #     kappa=kappa,
        # )
        self.decoder.log_var_branch = VarianceDecoderBranch(
            latent_dim=base_model.latent_dim,
            data=data,
            K=K,
            kappa=kappa,
            affine=True,
            p_dropout=0.1,
        )
        self.decoder.std_branch = LogVarToStd(self.decoder.log_var_branch)


class MLPVarianceDecoderBranch(nn.Module):
    # Maps latent_dim to MNIST dimension (1, 28, 28) through a MLP with RBF layer in the middle
    def __init__(
        self,
        latent_dim: int,
        data: Tensor,
        K: int,
        kappa: float = 1.0,
        hidden_layers: tuple[int, ...] = (128, 500),
    ):
        super().__init__()
        self.linear_rbf = LinearVarianceDecoderBranch(
            latent_dim=latent_dim, output_shape=64, data=data, K=K, kappa=kappa
        )
        self.layers = []
        for i, hidden_dim in enumerate(hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(64, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_dim))
            self.layers.append(nn.SiLU())
        self.layers.append(nn.Linear(hidden_layers[-1], 28 * 28))
        self.layers.append(nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)))
        self.log_var_branch = nn.Sequential(*self.layers)

    def forward(self, z: Tensor) -> Tensor:
        log_variance_hat = self.log_var_branch(z)
        return log_variance_hat


class StochasticVAE_MLP_RBF(BaseStochasticVAE):
    def __init__(
        self, latent_dim: int, base_model: VAE, data: Tensor, K: int, kappa: float = 1.0
    ):
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.decoder.mu_branch = base_model.decoder.mu_branch
        self.decoder.log_var_branch = MLPVarianceDecoderBranch(
            latent_dim=latent_dim, data=data, K=K, kappa=kappa, hidden_layers=(128, 500)
        )


#############################
# 3D Conv VAE models
#############################


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.SiLU(inplace=True)

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

        assert len(hidden_channels) == len(strides), (
            "hidden_channels and stride schedule mismatch"
        )

        blocks = []
        c_in = in_channels
        for c_out, s in zip(hidden_channels, strides):
            layers = [
                nn.Conv3d(c_in, c_out, kernel_size=3, stride=s, padding=1, bias=False),
                # nn.BatchNorm3d(c_out),
                nn.InstanceNorm3d(c_out, affine=True),
                nn.SiLU(inplace=True),
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
        #     nn.SiLU(inplace=True),
        #     nn.Linear(self.flat_dim // 2, latent_dim),
        # )
        # self.log_var_branch = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.flat_dim, self.flat_dim // 2),
        #     nn.SiLU(inplace=True),
        #     nn.Linear(self.flat_dim // 2, latent_dim),
        # )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
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

        assert len(hidden_channels) == len(upsample_sizes), (
            "hidden_channels and upsample schedule mismatch"
        )

        blocks = [nn.Unflatten(dim=-1, unflattened_size=bottleneck_shape)]
        c_in = c0
        for c_out, target_size in zip(hidden_channels, upsample_sizes):
            layers = [
                nn.Upsample(size=target_size, mode="trilinear", align_corners=False),
                nn.Conv3d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm3d(c_out),
                nn.InstanceNorm3d(c_out, affine=True),
                nn.SiLU(inplace=True),
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
        super().__init__()
        self.latent_dim = latent_dim
        # self.base_cometric = base_cometric

        self.layers = nn.Sequential()
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.add_module(f"linear_{i}", nn.Linear(self.latent_dim, dim))
            else:
                self.layers.add_module(f"linear_{i}", nn.Linear(hidden_dims[i - 1], dim))
            self.layers.add_module(f"silu_{i}", nn.SiLU())
        self.layers.add_module("linear_out", nn.Linear(hidden_dims[-1], self.latent_dim))

    def forward(self, x):
        omega = self.layers(x)
        # Compute |w(x)|_{base_cometric^{-1}(x)}
        # if not self.train:
        #     # We do that because otherwise the gradient
        #     # Is messed up during learning.
        #     # The normalization is not needed at training time
        #     omega_norm = self.base_cometric.cometric(x, omega)
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
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential()
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.add_module(f"linear_{i}", nn.Linear(self.latent_dim, dim))
            else:
                self.layers.add_module(f"linear_{i}", nn.Linear(hidden_dims[i - 1], dim))
            self.layers.add_module(f"silu_{i}", nn.SiLU())
        self.layers.add_module("linear_out", nn.Linear(hidden_dims[-1], 1))
        self.layers.add_module("sigmoid", nn.Sigmoid())

    def forward(self, z):
        return self.layers(z)


class DummyAccuracyPredictor(nn.Module):
    """
    Dummy accuracy predictor that always returns 1.0.
    This is used as a placeholder when no accuracy prediction is needed.
    """

    def __init__(self):
        super().__init__()

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
        omega_norm = self.cometric.cometric(X, omega)
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

    def __init__(
        self,
        in_channels: int = 1,
        out_feature: int = 1,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
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

        if output_activation is not None:
            self.act = output_activation
        else:
            self.act = nn.Identity()
        self.model = nn.Sequential(self.resize, self.resnet, self.act)

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
        dropout: float | None = None,
    ):
        super().__init__()
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

            self.fc.add_module(f"silu_{i}", nn.SiLU())
        self.fc.add_module("linear_out", nn.Linear(hidden_dim[-1], self.num_classes))

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
