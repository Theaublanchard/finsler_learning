import torch
from geodesic_toolbox import CoMetric, IdentityCoMetric, PullBackCometric
from pytorch_msssim import MS_SSIM
from torch import Tensor
from torchmetrics import MeanMetric
from torchquad import GaussLegendre


class Losse(torch.nn.Module):
    """
    General loss class.

    The forward method should return a tensor of shape (1,) containing the loss value,
    and set the attribute self.curr_it_component to a dictionary containing any component of the loss that should be logged during training.
    """

    def __init__(self):
        super().__init__()
        self.curr_it_component: dict[str, Tensor] = {}

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            "This is an abstract class, please implement the forward method."
        )


def margin_loss(
    omega_i: Tensor,
    p_i: Tensor,
    t_i: Tensor,
    p_j: Tensor,
    t_j: Tensor,
    margin: float = 0,
    cometric: CoMetric = IdentityCoMetric(),
    return_acc: bool = False,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """
    Computes the margin loss for the hyperplane predictor.
    The loss is defined as:
        L_ij = max(0, margin - w_i^T @ G_i @ (p_ij-p_i) * y_ij)
    where y_ij = 2 * (t_j < t_i) - 1 is the label of the hyperplane
    and where G_i is the cometric at point p_i.
    The loss is only computed on the points p_j that are defined for each p_i.
    ie p_j[i,:] are the K points to compute the margin loss for p_i[i].

    Parameters
    ----------
    omega_i : torch.Tensor (b,d)
        The predicted normal vector of the hyperplane
    p_i : torch.Tensor (b,d)
        The points at which the hyperplanes are defined.
    t_i : torch.Tensor (b,)
        The time variable of the points p_i.
    p_j : torch.Tensor (b,K,d)
        The points at which the hyperplanes are defined such that
        p_j[i,:] are the K points to compute the margin loss for p_i[i]
    t_j : torch.Tensor (b,K)
        The time variable of the points p_j.
    margin : float
        The margin to use for the loss.
    cometric : CoMetric
        The cometric to use for the dot product. Default is IdentityCoMetric.
    return_acc : bool
        If True, returns the accuracy of the hyperplane predictor.

    Returns
    -------
    loss : torch.Tensor (1,)
        The margin loss for each hyperplane.
    accuracy : torch.Tensor (b,)
        The accuracy of the hyperplane predictor.
    """
    # Sorry not good looking code
    # Needed to do optimisation

    assert 0 <= margin < 1

    B, K = t_j.shape

    loss_per_i = torch.empty(B, device=omega_i.device, dtype=omega_i.dtype)

    if return_acc:
        acc_per_i = torch.empty(B, device=omega_i.device, dtype=omega_i.dtype)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)

        omega = omega_i[start:end]
        p_i_chunk = p_i[start:end]

        # G = cometric.metric_tensor(p_i_chunk)
        # if G.dim() == 2:
        #     omega_raised = omega * G
        # elif G.dim() == 3:
        #     omega_raised = torch.einsum("bd,bde->be", omega, G)
        omega_raised = omega

        # omega^T G (p_j - p_i) = omega^T G p_j - omega^T G p_i
        # Avoid materializing delta_p = p_j - p_i[:, None]
        logits = torch.einsum("bd,bkd->bk", omega_raised, p_j[start:end])
        logits -= torch.einsum("bd,bd->b", omega_raised, p_i_chunk)[:, None]

        y_ij_chunk = t_j[start:end] < t_i[start:end, None]
        y_ij_chunk = 2 * y_ij_chunk.to(logits.dtype) - 1

        preloss = logits * y_ij_chunk

        loss_chunk = (1 - margin - preloss).clamp_min(0)
        loss_per_i[start:end] = loss_chunk.mean(dim=1)

        if return_acc:
            acc_per_i[start:end] = (preloss >= 0).to(logits.dtype).mean(dim=1)

    loss = loss_per_i.mean()

    if return_acc:
        return loss, acc_per_i

    return loss


def KNN(X_1: Tensor, X_2: Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the k nearest neighbors of X_1 in X_2.

    Parameters
    ----------
    X_1 : torch.Tensor (b,d)
        The points to compute the nearest neighbors for.
    X_2 : torch.Tensor (m,d)
        The points to compute the nearest neighbors from.
    k : int
        The number of nearest neighbors to compute.

    Returns
    -------
    knn : torch.Tensor (b,k,d)
        The nearest neighbors of X_1 in X_2.
    knn_idx : torch.Tensor (b,k)
        The indices of the nearest neighbors of X_1 in X_2.
    """
    dists = torch.cdist(X_1.detach(), X_2.detach())
    knn_idx = torch.argsort(dists, dim=1)[:, :k]
    knn = X_2[knn_idx]
    return knn, knn_idx


def knn_margin_loss(
    omega_i: Tensor,
    p_i: Tensor,
    t_i: Tensor,
    p_j: Tensor,
    t_j: Tensor,
    k: int,
    margin: str = 0,
    cometric=IdentityCoMetric(),
    return_acc: bool = False,
) -> Tensor:
    """Computes the margin loss for the hyperplane predictor.
    The loss is computed for each points only using its KNN.

    Parameters
    ----------
    omega_i : torch.Tensor (b,d)
        The predicted normal vector of the hyperplane
    p_i : torch.Tensor(b,d)
        The points at which the hyperplanes are defined.
    t_i : torch.Tensor(b,)
        The time variable of the points p_i.
    p_j : torch.Tensor (m,d)
        The points at which the hyperplanes are defined.
    t_j : torch.Tensor (m,)
        The time variable of the points p_j.
    k : int
        The number of neighbours to use.
    margin : float
        The margin to use for the loss.
    cometric : CoMetric
        The cometric to use for the dot product.
    return_acc : bool
        If True, returns the accuracy of the hyperplane predictor.

    Returns
    -------
    loss : torch.Tensor (1,)
        The margin loss for each hyperplane.
    accuracy : torch.Tensor (b,)
        The accuracy of the hyperplane predictor.
    """
    p_j, knn_idx = KNN(p_i, p_j, k)  # (b,k,d)
    t_j = t_j[knn_idx]  # (b,k)
    res = margin_loss(
        omega_i, p_i, t_i, p_j, t_j, margin=margin, cometric=cometric, return_acc=return_acc
    )
    return res


class GroupedMarginLoss(Losse):
    """
    Computes the margin loss for the hyperplane predictor
    with regards to a given set of points p_j and their time variable t_j.
    The dot product is computed with the cometric G_i at each point p_i.

    The loss is defined as:
    L_ij = max(0, margin - w_i^T @ G_i @ (p_j-p_i) * y_ij)

    where y_ij = 2 * (t_j < t_i) - 1 is the label of the hyperplane.

    Parameters
    ----------
    k : int, optional
        The number of nearest neighbors to use for the loss computation.
        If None, the loss is computed for all pairs of points.
    margin : float, optional
        The margin to use for the loss. Default is 0.0.
    """

    def __init__(self, k: int | None = None, margin: float = 0.0):
        super().__init__()
        self.k = k
        self.margin = margin
        self.acc_metric = MeanMetric()

    def forward(
        self,
        omega_i: Tensor,
        p_i: Tensor,
        t_i: Tensor,
        p_j: Tensor,
        t_j: Tensor,
        cometric: CoMetric = IdentityCoMetric(),
    ) -> Tensor:
        """
        Parameters
        ----------
        omega_i : Tensor (b,d)
            The predicted normal vector of the hyperplane.
        p_i : Tensor (b,d)
            The points at which the hyperplanes are defined.
        t_i : Tensor (b,)
            The time variable of the points p_i.
        p_j : Tensor (m,d)
            The points at which the hyperplanes are defined.
        t_j : Tensor (m,)
            The time variable of the points p_j.
        cometric : CoMetric
            The cometric to use for the dot product. Default is IdentityCoMetric.

        Returns
        -------
        loss : Tensor (1,)
            The margin loss for each hyperplane.
        """
        if self.k is not None:
            loss, accuracy = knn_margin_loss(
                omega_i,
                p_i,
                t_i,
                p_j,
                t_j,
                self.k,
                self.margin,
                cometric=cometric,
                return_acc=True,
            )
        else:
            loss, accuracy = margin_loss(
                omega_i,
                p_i,
                t_i,
                p_j,
                t_j,
                margin=self.margin,
                cometric=cometric,
                return_acc=True,
            )
        self.acc_metric(accuracy)
        self.acc = accuracy
        self.curr_it_component = {"margin_loss": loss, "accuracy": self.acc_metric.compute()}
        return loss


class MarginLoss(GroupedMarginLoss):
    """Margin loss for the hyperplane predictor.
    This class only uses the given points to compute the loss.
    There is no reference to a given set of points.

    Parameters
    ----------
    k : int, optional
        The number of nearest neighbors to use for the loss computation.
    margin : float, optional
        The margin to use for the loss. Default is 0.0.
    """

    def __init__(self, k: int | None = None, margin: float = 0.0):
        super().__init__(k=k, margin=margin)

    def forward(self, omega_i, p_i, t_i, cometric=IdentityCoMetric()):
        return super().forward(omega_i, p_i, t_i, p_i, t_i, cometric=cometric)


class NaNLossException(Exception):
    pass


class KL_divergence(Losse):
    """Compute the KL divergence between a normal distribution and a standard normal distribution."""

    def __init__(self, free_bits: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.free_bits = free_bits

    def forward(
        self, mu: Tensor, log_var_latent: Tensor, return_components: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        mu : Tensor (B, D)
            The mean of the latent distribution.
        log_var_latent : Tensor (B, D)
            The log variance of the latent distribution.
        return_components : bool
            If True, returns a dictionary containing the computed KL divergence and its components. Default is False
        Returns
        -------
        kl : Tensor (1,)
            The KL divergence between the latent distribution and the standard normal distribution.
        if return_components is True, also returns a dictionary containing the computed KL divergence and its components.
        """
        normal_0 = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * log_var_latent))
        normal_1 = torch.distributions.Normal(
            loc=torch.zeros_like(mu), scale=torch.ones_like(log_var_latent)
        )

        kl_per_dim_raw = torch.distributions.kl_divergence(normal_0, normal_1)
        kl_raw = kl_per_dim_raw.sum(-1).mean()
        kl_per_dim = kl_per_dim_raw.sum(-1)
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        kl = kl_per_dim.mean()

        self.curr_it_component = {
            "kl": kl,
            "kl_raw": kl_raw,
            "kl_free_bits_delta": kl - kl_raw,
        }
        if not return_components:
            return kl
        return kl, self.curr_it_component


class KL_divergence_spCauchy(Losse):
    """Computes the KL divergence between a spherical Cauchy distribution and a uniform distribution on the unit hypersphere.

    We use the method from 'Hyperspherical Variational Autoencoders Using Efficient Spherical Cauchy Distribution'

    Parameters
    ----------
    N : int
            The number of quadrature points to use for the integral approximation.
    threshold : float
            The threshold value for the concentration parameter to switch between the approximation and the integral formulation.
    free_bits : float
            Not used in this implementation, but kept for consistency with other loss classes.
    """

    def __init__(
        self, N: int = 100, threshold: float = 0.9, free_bits: float = 0.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.integrator = GaussLegendre()
        self.N = N
        self.threshold = threshold
        self.free_bits = free_bits

    def kl_approximation(self, mu: Tensor, rho: Tensor) -> Tensor:
        """
        Computes the KL divergence between a spherical Cauchy distribution and a uniform distribution on the unit hypersphere using the approximation from Prop2 in the paper.
        This is valid for high concentration values (rho > 0.9).
        KL = (d-1) * ( log((1+rho)/(1-rho)) + log(Gamma((d-1)/2)) - log(Gamma(d-1)) )

        Parameters
        ----------
        mu : Tensor (B, D)
            The mean of the spherical Cauchy distribution.
        rho : Tensor (B, 1)
            The concentration parameter of the spherical Cauchy distribution.

        Returns
        -------
        kl : Tensor (1,)
            The KL divergence between the spherical Cauchy distribution and the uniform distribution.
        """
        d = mu.shape[1]
        d_half = torch.tensor((d - 1) / 2, device=mu.device, dtype=mu.dtype)
        d_minus_1 = torch.tensor(d - 1, device=mu.device, dtype=mu.dtype)
        kl_approx = (d - 1) * (torch.log1p(rho) - torch.log1p(-rho))
        kl_approx += (d - 1) * (torch.digamma(d_half) - torch.digamma(d_minus_1))
        return kl_approx.squeeze(-1)

    def kl_integral(self, mu: Tensor, rho: Tensor) -> Tensor:
        """
        Computes the KL divergence between a spherical Cauchy distribution and a uniform distribution on the unit hypersphere using the integral formulation from Prop1 in the paper.
        This is valid for low concentration values (rho <= 0.9).
        KL = (d-1) log((1-rho)/(1+rho)) + (d-1) int_0^1 t^(d-2)/(1-t) ( 1 - ((1-rho^2)^/((1+rho^2)-4t rho))^(d-1)/2 ) dt

        Parameters
        ----------
        mu : Tensor (B, D)
            The mean of the spherical Cauchy distribution.
        rho : Tensor (B, 1)
            The concentration parameter of the spherical Cauchy distribution.

        Returns
        -------
        kl : Tensor (1,)
            The KL divergence between the spherical Cauchy distribution and the uniform distribution.
        """
        d = mu.shape[1]
        cst = (d - 1) * (torch.log1p(-rho) - torch.log1p(rho))

        def integrand(t, rho):
            t = t.unsqueeze(-1)
            # rho = rho.unsqueeze(0)
            # very ugly IO overhead but torchquad does weird stuff with generators
            rho = rho.unsqueeze(0).to(t.device).to(t.dtype)
            alpha = (d - 1) / 2
            denom = (1 + rho).square() - 4 * t * rho
            log_ratio = 2 * torch.log1p(-rho) - torch.log(denom)
            # We use this because torch.pow(denom, -alpha) can be numerically unstable for small denom and large alpha.
            numerator = -torch.expm1(alpha * log_ratio)
            return t.pow(d - 2) * numerator / (1 - t)

        # data_type = "float32" if mu.dtype == torch.float32 else "float64"
        # is_gpu = mu.device.type == "cuda"
        # set_up_backend("torch",data_type=data_type, torch_enable_cuda=is_gpu)
        integrator = GaussLegendre()
        integral = (
            integrator.integrate(
                integrand, dim=1, N=self.N, integration_domain=[[0, 1]], args=(rho,)
            )
            .to(mu.device)
            .to(mu.dtype)
        )
        kl_integral = cst + (d - 1) * integral
        return kl_integral.squeeze(-1)

    def forward(
        self, mu: Tensor, rho: Tensor, return_components: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Computes the KL divergence between a spherical Cauchy distribution and a uniform distribution on the unit hypersphere.
        For high concentration value we use the approximation from Prop2.
        Otherwise, we use the integral formulation and compute using quadrature integration.

        Parameters
        ----------
        mu : Tensor (B, D)
            The mean of the spherical Cauchy distribution.
        rho : Tensor (B, 1)
            The concentration parameter of the spherical Cauchy distribution.

        Returns
        -------
        kl : Tensor (1,)
            The KL divergence between the spherical Cauchy distribution and the uniform distribution.
        if return_components is True, also returns a dictionary containing the computed KL divergence and its components.
        """
        kl = torch.zeros(mu.shape[0], device=mu.device, dtype=mu.dtype)

        approx_idx = (rho > self.threshold).squeeze(-1)
        rho_approx = rho[approx_idx]
        rho_integral = rho[~approx_idx]
        if rho_approx.numel() > 0:
            kl[approx_idx] = self.kl_approximation(mu[approx_idx], rho_approx)
        if rho_integral.numel() > 0:
            kl[~approx_idx] = self.kl_integral(mu[~approx_idx], rho_integral)

        self.curr_it_component = {"kl": kl.mean()}
        if not return_components:
            return kl.mean()
        return kl.mean(), self.curr_it_component


class LogPrior(Losse):
    """
    Compute the log likelyhood of a multivariate normal distribution.
    In the usual settings, this amounts to the reconstruction loss of a VAE model.
    Ie, the L2 norm between the reconstructed input and the original input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, x_hat: Tensor, x: Tensor, log_var_data: Tensor, return_components: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        x_hat : Tensor (B, ...)
            The reconstructed input tensor.
        x : Tensor (B, ...)
            The original input tensor.
        log_var_data : Tensor (B, ...)
            The log variance of the data distribution.
        return_components : bool
            If True, returns a dictionary containing the computed log likelyhood and its components. Default is False

        Returns
        -------
        neg_log_prior : Tensor (1,)
            The negative log likelyhood of the data distribution.
        if return_components is True, also returns a dictionary containing the computed negative log likelyhood and its components.
        """
        normal = torch.distributions.Normal(
            loc=x_hat.flatten(start_dim=1),
            scale=torch.exp(0.5 * log_var_data.flatten(start_dim=1)),
        )
        log_prior = normal.log_prob(x.flatten(start_dim=1))  # (B, D)
        log_prior = log_prior.sum(-1).mean()  # (1,)
        if not return_components:
            return -log_prior
        else:
            self.curr_it_component = {"neg_log_prior": -log_prior}
            return -log_prior, self.curr_it_component


class VAELoss(Losse):
    """Computes the ELBO loss for a VAE model. Optionnal beta parameter to weight the KL divergence."""

    def __init__(self, beta: float = 1.0, free_bits: float = 0.0):
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits
        self.kl = KL_divergence(free_bits=free_bits)
        self.prior = LogPrior()

    def forward(
        self,
        x_hat: Tensor,
        x: Tensor,
        mu: Tensor,
        log_var_latent: Tensor,
        log_var_data: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        x_hat : Tensor (B, ...)
            The reconstructed input tensor.
        x : Tensor (B, ...)
            The original input tensor.
        mu : Tensor (B, D)
            The mean of the latent distribution.
        log_var_latent : Tensor (B, D)
            The log variance of the latent distribution.
        log_var_data : Tensor (B, ...)
            The log variance of the data distribution.
        return_components : bool
            If True, returns a dictionary containing the computed loss components (reconstruction loss and KL divergence).

        Returns
        -------
        loss : Tensor (1,)
            The computed loss value, which is the sum of the reconstruction loss and the KL divergence.
        if return_components is True, also returns a dictionary containing the computed loss components (reconstruction loss and KL divergence).
        """
        neg_log_prior, prior_components = self.prior(
            x_hat, x, log_var_data, return_components=True
        )
        kl_, kl_components = self.kl(mu, log_var_latent, return_components=True)
        loss = neg_log_prior + self.beta * kl_
        self.curr_it_component = (
            {"base_vae_loss": loss} | self.kl.curr_it_component | self.prior.curr_it_component
        )
        if not return_components:
            return loss
        return loss, self.curr_it_component


class VAELossspCauchy(Losse):
    """Computes the ELBO loss for a VAE model. Optionnal beta parameter to weight the KL divergence."""

    def __init__(self, beta: float = 1.0, free_bits: float = 0.0):
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits
        if free_bits > 0:
            print("Free bits not supported for spherical Cauchy KL divergence.")
        self.kl = KL_divergence_spCauchy(free_bits=free_bits)
        self.prior = LogPrior()

    def forward(
        self,
        x_hat: Tensor,
        x: Tensor,
        mu: Tensor,
        rho: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        x_hat : Tensor (B, ...)
            The reconstructed input tensor.
        x : Tensor (B, ...)
            The original input tensor.
        mu : Tensor (B, D)
            The mean of the latent distribution on the unit sphere.
        rho : Tensor (B, D)
            The rho parameter of the latent distribution in [0, 1).
        return_components : bool
            If True, returns a dictionary containing the computed loss components (reconstruction loss and KL divergence).

        Returns
        -------
        loss : Tensor (1,)
            The computed loss value, which is the sum of the reconstruction loss and the KL divergence.
        if return_components is True, also returns a dictionary containing the computed loss components (reconstruction loss and KL divergence).
        """
        log_var_data = torch.zeros_like(
            x_hat
        )  # Assuming unit variance for the data distribution
        neg_log_prior, prior_components = self.prior(
            x_hat, x, log_var_data, return_components=True
        )
        kl_, kl_components = self.kl(mu, rho, return_components=True)
        loss = neg_log_prior + self.beta * kl_
        self.curr_it_component = (
            {"base_vae_loss": loss} | self.kl.curr_it_component | self.prior.curr_it_component
        )
        if not return_components:
            return loss
        return loss, self.curr_it_component


class MS_SSIMLoss(Losse):
    """Multi-Scale Structural Similarity (MS-SSIM) loss function.
    Code ~from https://github.com/VainF/pytorch-msssim

    Parameters:
    -----------
    - data_range: The range of the input data. Default is 1.0.
    - size_average: If True, the loss is averaged over each loss element in the batch. Default is True.
    - n_channels: Number of channels in the input. Default is 3.
    - ndim: Number of dimensions of the input data (2D or 3D). Default is 2.
    """

    def __init__(
        self, data_range: float = 1.0, size_average: bool = True, n_channels: int = 3, ndim=2
    ):
        super().__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.n_channels = n_channels
        self.msssim = MS_SSIM(
            data_range=data_range,
            size_average=size_average,
            channel=n_channels,
            spatial_dims=2 if (ndim == 2) else 3,
        )
        if ndim == 2:
            self.resize = torch.nn.Upsample(size=(161, 161))
        elif ndim == 3:
            self.resize = torch.nn.Upsample(size=(161, 161, 161))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, return_components: bool = False
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (B, C, H, W) or (B, C, D, H, W)
            The first input tensor.
        y : torch.Tensor (B, C, H, W) or (B, C, D, H, W)
            The second input tensor, which should have the same shape as x.
        return_components : bool
            If True, returns a dictionary containing the computed MS-SSIM value and the loss value separately. Default is False.
        Returns
        -------
        loss : torch.Tensor (1,)
            The computed MS-SSIM loss between the two input tensors.
        """
        if x.shape != y.shape:
            raise ValueError("Input tensors must have the same shape.")
        if (torch.tensor(x.shape[2:]) < 160).any():
            # Resize the input tensors if their spatial dimensions are less than 160
            x = self.resize(x)
            y = self.resize(y)
        loss = 1 - self.msssim(x, y)
        self.curr_it_component = {"mssim_loss": loss}
        if not return_components:
            return loss
        else:
            return loss, self.curr_it_component


class FlatManifoldLoss(Losse):
    """
    Loss from "Learning Flat Latent Manifolds with VAEs".
    It pull the decoder towards a the scaled identity metric.

    Parameters
    ----------
    decoder : torch.nn.Module
        The decoder of the VAE model.
    n_steps : int
        The number of steps to use for the linear interpolation between pairs of points in the latent space
    """

    def __init__(self, decoder: torch.nn.Module, n_steps: int = 10):
        super().__init__()
        self.decoder = decoder
        self.pb_dec = PullBackCometric(diffeo=self.decoder, method="finite_difference")
        self.c_square = 1.0
        self.n_steps = n_steps

    def linear_interpolation(self, z_0: Tensor, z_1: Tensor, n_steps: int) -> Tensor:
        """
        Parameters
        ----------
        z_0 : Tensor (B, D)
            The starting point of the interpolation.
        z_1 : Tensor (B, D)
            The ending point of the interpolation.
        n_steps : int
            The number of steps in the interpolation.

        Returns
        -------
        z_interp : Tensor (B *  n_steps, D)
            The interpolated points between z_0 and z_1.
        """
        B, D = z_0.shape
        t = torch.linspace(0, 1, n_steps, device=z_0.device)
        t = t.view(1, n_steps, 1).expand(B, n_steps, D)
        z_interp = z_0.unsqueeze(1) * (1 - t) + z_1.unsqueeze(1) * t
        z_interp = z_interp.view(B * n_steps, D)
        return z_interp

    def set_c_square(self, z: Tensor) -> None:
        """
        Compute c^2 = mean(Tr(G(z_inter))) where G is the pullback metric of the decoder.

        Here z_inter is a set of interpolated points between pairs of points in z.

        Parameters
        ----------
        z : Tensor (B, D)
            The latent variable tensor.

        Returns
        -------
        c_square : float
            The computed scaling factor for the identity metric.
        """
        idx = torch.randperm(z.shape[0])
        z_0 = z[idx[: z.shape[0] // 2]]
        z_1 = z[idx[z.shape[0] // 2 :]]
        z_inter = self.linear_interpolation(z_0, z_1, n_steps=self.n_steps)
        G = self.pb_dec.metric_tensor(z_inter)
        c_square = torch.trace(G).mean().item()
        self.c_square = c_square

    def forward(self, z: Tensor, return_components: bool = False) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor (B, D)
            The latent variable tensor.
        return_components : bool
            If True, returns a dictionary containing the computed loss value and its components. Default is False.

        Returns
        -------
        loss : Tensor (1,)
            The computed loss value, which is the mean squared Frobenius norm of the difference between the pullback metric and the scaled identity metric.
        """
        self.set_c_square(z)
        G = self.pb_dec.metric_tensor(z)
        G_minus_id = G - self.c_square * self.pb_dec.eye(z)  # (B, D, D)
        loss = torch.linalg.norm(G_minus_id, dim=(-2, -1)) ** 2
        loss = loss.mean()
        self.curr_it_component = {"flat_manifold_loss": loss}
        if not return_components:
            return loss
        else:
            return loss, self.curr_it_component
