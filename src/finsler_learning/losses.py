import torch
from torch import Tensor
from pytorch_msssim import MS_SSIM
from torchmetrics import MeanMetric
from tqdm import tqdm
from geodesic_toolbox import CoMetric, IdentityCoMetric
from einops import rearrange, repeat


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
    reduce: str = "mean",
    margin: float = 0,
    cometric: CoMetric = IdentityCoMetric(),
    return_acc: bool = False,
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
    reduce : str
        The reduction method to use. Can be 'mean' or 'sum'.
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
    assert reduce in ["mean", "sum"], f"reduce should be in ['mean', 'sum'], but got {reduce}"
    assert 1 > margin >= 0, f"margin should be in [0,1), but got {margin}"

    y_ij = 2 * (t_j < t_i[:, None]) - 1  # (b,K)
    delta_p = p_j - p_i[:, None, :]  # (b,K,d)

    omega_i_ = repeat(omega_i, "b d -> b K d", K=delta_p.shape[1])
    p_i_ = repeat(p_i, "b d -> b K d", K=delta_p.shape[1])
    omega_i_ = rearrange(omega_i_, "b K d -> (b K) d")
    p_i_ = rearrange(p_i_, "b K d -> (b K) d")
    delta_p_ = rearrange(delta_p, "b K d -> (b K) d")
    y_ij_ = rearrange(y_ij, "b K -> (b K)")

    # The vectorized dot path can be prohibitively large for pullback cometrics.
    # Stream the flattened pairwise inputs to the cometric in small chunks so each
    # cometric.dot call only sees a bounded number of samples.
    b = omega_i.shape[0]
    k = delta_p.shape[1]
    chunk_size = 128

    loss_sum = torch.zeros(b, device=omega_i.device, dtype=omega_i.dtype)
    acc_sum = torch.zeros(b, device=omega_i.device, dtype=omega_i.dtype)

    total_pairs = b * k
    for start in tqdm(range(0, total_pairs, chunk_size), desc="Computing margin loss",leave=False):
        end = min(start + chunk_size, total_pairs)
        omega_chunk = omega_i_[start:end]
        p_chunk = p_i_[start:end]
        v_chunk = delta_p_[start:end]
        y_chunk = y_ij_[start:end]

        logits_chunk = cometric.dot(p_chunk, omega_chunk, v_chunk)
        preloss_chunk = logits_chunk * y_chunk
        loss_chunk = (1 - margin - preloss_chunk).clamp(0)

        loss_sum += loss_chunk.sum(dim=0)
        acc_sum += (preloss_chunk >= 0).to(preloss_chunk.dtype).sum(dim=0)

    if reduce == "mean":
        loss = loss_sum / k
        acc = acc_sum / k  # (b,)
    elif reduce == "sum":
        loss = loss_sum
        acc = acc_sum  # (b,)

    if not return_acc:
        return loss.mean()
    else:
        return loss.mean(), acc


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
        omega_i,
        p_i,
        t_i,
        p_j,
        t_j,
        reduce="mean",
        margin=margin,
        cometric=cometric,
        return_acc=return_acc,
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

    def __init__(self, k: int = None, margin: float = 0.0):
        super(GroupedMarginLoss, self).__init__()
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

    def __init__(self, k: int = None, margin: float = 0.0):
        super(MarginLoss, self).__init__(k=k, margin=margin)

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
            self.curr_it_component = {
                "neg_log_prior": -log_prior,
            }
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
            {
                "base_vae_loss": loss,
            }
            | self.kl.curr_it_component
            | self.prior.curr_it_component
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
