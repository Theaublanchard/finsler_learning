import matplotlib.pyplot as plt
import torch
import logging
from dataclasses import dataclass
from datetime import datetime
from torch import Tensor
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from geomloss import SamplesLoss
from .logger import setup_logger
from .models import BaseEncoder, BaseDecoder
from .base_geodesic_evaluation import retrieve_time_from_traj
from .base_geodesic_evaluation import ALL_METRICS
from .base_extrapolation import (
    export_extrapolation_evaluation,
    get_metrics_summary,
    export_extrapolated_trajectories,
    export_extrapolated_time,
    filter_traj_nan,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class WassersteinConfig:
    export_path: Path
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    encoder: BaseEncoder
    decoder: BaseDecoder
    regressor: torch.nn.Module
    latent_dim: int = 8
    n_bins: int = 10
    lr: float = 1e-2
    n_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    T: int = 100


@torch.no_grad()
def embed_dataset(dataloader, encoder, device) -> tuple[Tensor, Tensor]:
    """
    Embeds a dataset using the provided encoder.

    Parameters:
    -----------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the dataset to embed
    encoder : torch.nn.Module
        The encoder to use for embedding
    device : torch.device
        The device to use for embedding

    Returns:
    --------
    z_embedded : torch.Tensor (N,D)
        The embedded dataset where N is the number of samples and D is the latent dimension
    t_embedded : torch.Tensor (N,)
        The corresponding time points for each embedded sample
    """
    z_embedded = []
    t_embedded = []
    for x, t in tqdm(dataloader):
        x = x.to(device)
        z = encoder(x)[0]
        z_embedded.append(z.cpu())
        t_embedded.append(t.cpu())
    z_embedded = torch.cat(z_embedded, dim=0)
    t_embedded = torch.cat(t_embedded, dim=0)
    return z_embedded, t_embedded


class Potential(torch.nn.Module):
    """
    A simple feedforward neural network to model the potential function in the latent space.
    The network takes a latent vector as input and outputs a scalar potential value.
    It also implements a method to compute the gradient of the potential with respect to the input latent vector.
    """

    def __init__(self, in_dim: int, hidden_dims: list = [128]):
        super().__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(torch.nn.Linear(in_dim, hidden_dims[i]))
            layers.append(torch.nn.GELU())
            in_dim = hidden_dims[i]
        self.net = torch.nn.Sequential(*layers)
        self.net.append(
            torch.nn.Linear(hidden_dims[-1], 1)
        )  # Output layer for potential function

    def potential(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the potential function for a given latent vector z.

        Parameters:
        -----------
        z : torch.Tensor (B, D)
            The input latent vectors where B is the batch size and D is the latent dimension.

        Returns:
        --------
        potential : torch.Tensor (B,)
            The scalar potential value for each input latent vector.
        """
        return self.net(z).squeeze(-1)  # Return a scalar potential for each input

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the potential function with respect to the input latent vector z.

        Parameters:
        -----------
        z : torch.Tensor (B, D)
            The input latent vectors where B is the batch size and D is the latent dimension.

        Returns:
        --------
        grad : torch.Tensor (B, D)
            The gradient of the potential function with respect to z.
        """
        z.requires_grad_(True)
        potential = self.potential(z)
        grad = torch.autograd.grad(
            outputs=potential,
            inputs=z,
            grad_outputs=torch.ones_like(potential),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad

    def pushforward(self, z: torch.Tensor, tau: torch.Tensor) -> Tensor:
        """
        Computes the pushforward of the latent vector z under the potential function for a given time step tau.

        Parameters:
        -----------
        z : torch.Tensor (B, D)
            The input latent vectors where B is the batch size and D is the latent dimension.
        tau : torch.Tensor (B,)
            The time steps for the pushforward operation.

        Returns:
        --------
        z_pushforward : torch.Tensor (B, D)
            The pushed forward latent vectors after applying the potential function.
        """
        grad = self.forward(z)
        z_pushforward = z - tau[:, None] * grad  # Update rule for pushforward
        return z_pushforward


def frequence_bins(t, n_bins: int = 10):
    """
    Bins the time points into equal frequency bins.

    Parameters:
    -----------
    t : torch.Tensor (N,)
        The time points to bin
    n_bins : int
        The number of bins to create

    Returns:
    --------
    bin_edges : torch.Tensor (n_bins+1,)
        The edges of the bins
    """
    quantiles = torch.linspace(0, 1, n_bins + 1).to(t.dtype)
    bin_edges = torch.quantile(t, quantiles)
    return bin_edges


def to_bin(z, t, bin_edges):
    """
    Assigns each latent vector z to a bin based on its corresponding time point t.

    Parameters:
    -----------
    z : torch.Tensor (N,D)
        The latent vectors
    t : torch.Tensor (N,)
        The corresponding time points for each latent vector
    bin_edges : torch.Tensor (n_bins+1,)
        The edges of the bins

    Returns:
    --------
    binned_z : list[torch.Tensor]
        A list of tensors where each tensor contains the latent vectors assigned to that bin
    """
    # Assign each time point to a bin
    bin_indices = torch.bucketize(t, bin_edges, right=True) - 1  # -1 to make it 0-indexed

    # Create a list of tensors for each bin
    binned_z = []
    for i in range(len(bin_edges) - 1):
        binned_z.append(z[bin_indices == i])

    return binned_z


def global_loss(
    z: torch.Tensor,
    t: torch.Tensor,
    J: Potential,
    bin_edges: torch.Tensor,
    loss: SamplesLoss,
    device: torch.device,
):
    """
    Compute the loss as :
    L = sum_{i=0}^{n_bins-1} bin_width_i * W_2^2(p(z_i), mu(z_{i+1}))
    where p(z_i) = Id - tau * grad(J(z_i)) and mu(z_{i+1}) is the empirical distribution of z_{i+1}.

    Parameters:
    -----------
    z : torch.Tensor (N,D)
        The latent vectors
    t : torch.Tensor (N,)
        The corresponding time points for each latent vector
    J : Potential
        The potential function to compute the pushforward
    bin_edges : torch.Tensor (n_bins+1,)
        The edges of the bins
    loss : SamplesLoss
        The loss function to compute the Wasserstein distance
    device : torch.device
        The device to perform computations on
    """
    binned_z = to_bin(z, t, bin_edges)
    total_loss = 0.0
    for i in range(len(binned_z) - 1):
        z_i = binned_z[i]
        z_next = binned_z[i + 1]
        if z_i.shape[0] == 0 or z_next.shape[0] == 0:
            continue
        z_i = z_i.to(device).to(torch.float64)
        z_next = z_next.to(device).to(torch.float64)
        bin_width = bin_edges[i + 1] - bin_edges[i]  # (1,)

        bin_width = bin_width.repeat(z_i.shape[0]).to(device).to(torch.float64)  # (B,)
        z_i_pushforward = J.pushforward(z_i, tau=bin_width)
        w2_loss = loss(z_i_pushforward, z_next)
        total_loss += bin_width[0] * w2_loss
    return total_loss


def train_potential(
    z_train: torch.Tensor,
    t_train: torch.Tensor,
    z_test: torch.Tensor,
    t_test: torch.Tensor,
    config: WassersteinConfig,
):

    J = Potential(
        in_dim=config.latent_dim, hidden_dims=[config.latent_dim, config.latent_dim]
    ).to(config.device)
    loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
    bin_edges = frequence_bins(t_train, n_bins=config.n_bins)

    optim = torch.optim.Adam(J.parameters(), lr=config.lr)

    losses_train = []
    losses_test = []
    pbar = tqdm(range(config.n_epochs), desc="Training Potential J")
    for i in pbar:
        optim.zero_grad()
        loss_train = global_loss(z_train, t_train, J, bin_edges, loss_fn, device=config.device)
        loss_train.backward()
        optim.step()

        loss_test = global_loss(z_test, t_test, J, bin_edges, loss_fn, device=config.device)
        pbar.set_postfix({"loss": loss_train.item(), "loss_test": loss_test.item()})
        losses_train.append(loss_train.item())
        losses_test.append(loss_test.item())

        if i % 10 == 0:
            plot_export_losses(losses_train, losses_test, config.export_path)
    return J, losses_train, losses_test


def plot_export_losses(losses_train, losses_test, export_path: Path):
    """
    Plot and export the training and test losses of the potential function J.

    Parameters:
    -----------
    losses_train : list
        The training losses
    losses_test : list
        The test losses
    export_path : Path
        The path to export the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(losses_train, label="Train Loss")
    ax.plot(losses_test, label="Test Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Test Loss of Potential J")
    ax.legend()
    if (losses_train > 0).all() and (losses_test > 0).all():
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(export_path / "losses_plot.png")


def export_potential_training_results(J, losses_train, losses_test, config: WassersteinConfig):
    """
    Export the training results of the potential function J.

    Parameters:
    -----------
    J : Potential
        The trained potential function
    losses_train : list
        The training losses
    losses_test : list
        The test losses
    config : WassersteinConfig
        The configuration for training the potential function
    """
    losses_train = torch.tensor(losses_train)
    losses_test = torch.tensor(losses_test)

    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    torch.save(J.state_dict(), export_path / "potential_J.pth")
    torch.save(losses_train, export_path / "losses_train.pth")
    torch.save(losses_test, export_path / "losses_test.pth")

    plot_export_losses(losses_train, losses_test, export_path)


def sample_t_max(t_train: torch.Tensor, t_0: torch.Tensor):
    """
    For each time point in t_0,
    find the maximum time point in t_train that is greater than or equal to it.
    Then sample a value in the range [t_0, t_max] for each time point in t_0.

    Parameters:
    -----------
    t_train : torch.Tensor (N,)
        The training time points
    t_0 : torch.Tensor (M,)
        The time points to find the maximum for

    Returns:
    --------
    t_max : torch.Tensor (M,)
        The maximum time points in t_train for each time point in t_0
    """
    t_max = []
    for t in t_0:
        t_max_i = t_train[t_train >= t].max()
        t_max.append(t_max_i)
    t_max = torch.stack(t_max).to(t_0.device)
    # return t_max
    t_sampled = []
    for t, t_max_i in zip(t_0, t_max):
        t_sampled_i = torch.rand(1, device=t_0.device) * (t_max_i - t) + t
        t_sampled.append(t_sampled_i)
    t_sampled = torch.stack(t_sampled).squeeze()
    return t_sampled


def extrapolate_J(
    J: Potential, z_0: torch.Tensor, t_0: torch.Tensor, t_target: torch.Tensor, T: int
):
    """
    Extrapolate the latent vector z_0 to the target time t_target using the potential function J.

    Parameters:
    -----------
    J : Potential
        The trained potential function
    z_0 : torch.Tensor (B,D)
        The initial latent vector
    t_0 : torch.Tensor (B,)
        The initial time point
    t_target : torch.Tensor (B,)
        The target time point for extrapolation
    T : int
        The number of time steps for extrapolation

    Returns:
    --------
    z_traj : torch.Tensor (B,T,D)
        The extrapolated latent trajectories for each sample in the batch
    """
    tau = (t_target - t_0) / (T - 1)
    tau = tau.to(z_0.device).to(torch.float64)

    z_traj = [z_0]
    z_current = z_0
    for _ in range(T - 1):
        z_next = J.pushforward(z_current, tau)
        z_traj.append(z_next)
        z_current = z_next

    z_traj = torch.stack(z_traj, dim=1)  # (B,T,D)
    return z_traj


def compute_extrapolation_and_monotonicity(
    config: WassersteinConfig,
    dataloader: torch.utils.data.DataLoader,
    J: Potential,
    t_train: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Compute the extrapolated trajectories and evaluate monotonicity metrics.

    Parameters:
    -----------
    config : WassersteinConfig
        The configuration for training the potential function
    dataloader : torch.utils.data.DataLoader
        The dataloader for the dataset to evaluate
    J : Potential
        The trained potential function
    t_train : torch.Tensor (N,)
        The training time points used for sampling the target time points

    Returns:
    --------
    traj : torch.Tensor (N,T,D)
        The extrapolated trajectories for all samples in the dataset
    res : dict[str, torch.Tensor]
        A dictionary containing the computed metrics for monotonicity evaluation
    t_pred : torch.Tensor (N,T)
        The predicted time for each point in the extrapolated trajectories
    """
    J.eval()

    res: dict[str : list[torch.Tensor]] = {name: [] for name in ALL_METRICS.keys()}
    all_trajs = []
    all_t_pred = []

    pbar = tqdm(
        enumerate(dataloader),
        desc=f"Extrapolation W2 evaluation",
        total=len(dataloader),
        unit="batch",
    )
    for i, data in pbar:
        x, t_0 = data
        x = x.to(config.device)
        z_0 = config.encoder(x)[0].to(torch.float64)
        t_target = sample_t_max(t_train=t_train, t_0=t_0)
        traj = extrapolate_J(J, z_0, t_0, t_target, T=config.T)
        traj = filter_traj_nan(traj)

        if traj is None:
            continue

        all_trajs.append(traj.cpu())

        t_pred = retrieve_time_from_traj(
            traj.to(torch.float32),
            decoder=config.decoder,
            regressor=config.regressor,
            use_pbar=False,
        )  # (B,T)

        all_t_pred.append(t_pred.cpu())

        for name, metric_fn in ALL_METRICS.items():
            mono_measure = metric_fn(t_pred)  # (B,)
            res[name].append(mono_measure.cpu())

    all_trajs = torch.cat(all_trajs, dim=0)  # (N_samples,T,d)
    all_t_pred = torch.cat(all_t_pred, dim=0)  # (N_samples,T)

    for name in ALL_METRICS.keys():
        res[name] = torch.cat(res[name], dim=0)  # (N_samples,)

    return all_trajs, res, all_t_pred


def extrapolation_evaluation_gradient_flow(
    config: WassersteinConfig,
    J: Potential,
    t_train: torch.Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Perform extrapolation evaluation using the provided configuration.

    Parameters
    ----------
    config : ExtrapolationEvalRiemannConfig
        Configuration object containing all necessary parameters and models.
    J : Potential
        The potential function to use for extrapolation.
    t_train : torch.Tensor
        The training time points.

    Returns
    -------
    traj_test : torch.Tensor (N_pts*n_runs,T,d)
        The extrapolated trajectories for all samples in the test set.
    traj_train : torch.Tensor (N_pts*n_runs,T,d)
        The extrapolated trajectories for all samples in the train set.
    t_pred_test : torch.Tensor (N_pts*n_runs,T)
        The predicted time for each point in the extrapolated trajectories for the test set.
    t_pred_train : torch.Tensor (N_pts*n_runs,T)
        The predicted time for each point in the extrapolated trajectories for the train set.
    """
    start_time = datetime.now()

    LOGGER.info("Computing extrapolation and monotonicity for test set...")
    traj_test, res_test, t_pred_test = compute_extrapolation_and_monotonicity(
        config, config.test_loader, J, t_train
    )

    LOGGER.info("Computing extrapolation and monotonicity for train set...")
    traj_train, res_train, t_pred_train = compute_extrapolation_and_monotonicity(
        config, config.train_loader, J, t_train
    )

    summary_test = get_metrics_summary(res_test)
    summary_train = get_metrics_summary(res_train)
    LOGGER.info("Extrapolation evaluation results:")
    for name in ALL_METRICS.keys():
        LOGGER.info(
            f"{name.upper()}"
            f" - Test: mean={summary_test[name]['mean']:.4f} +- {summary_test[name]['std']:.4f}, "
            f" - Train: mean={summary_train[name]['mean']:.4f} +- {summary_train[name]['std']:.4f}"
        )

    LOGGER.info("Exporting results...")
    export_extrapolation_evaluation(
        config,
        res_test,
        res_train,
    )
    export_extrapolated_trajectories(
        config,
        traj_test,
        traj_train,
    )
    export_extrapolated_time(
        config,
        t_pred_test,
        t_pred_train,
    )

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    LOGGER.info(
        f"Extrapolation evaluation completed in {elapsed_time} for {config.export_path}"
    )
    return traj_test, traj_train, t_pred_test, t_pred_train


def main(config: WassersteinConfig):
    """
    Main function to train the potential function J on the embedded dataset.

    Parameters:
    -----------
    config : WassersteinConfig
        The configuration for training the potential function
    """

    # Setup LOGGER
    LOGGER = setup_logger(
        name=None,
        log_file=config.export_path / "pipeline.log",
        level=logging.INFO,
    )

    # Embed the training and test datasets
    z_train, t_train = embed_dataset(config.train_loader, config.encoder, device=config.device)
    z_test, t_test = embed_dataset(config.test_loader, config.encoder, device=config.device)

    # Train the potential function J
    J, losses_train, losses_test = train_potential(z_train, t_train, z_test, t_test, config)

    # Export the training results
    export_potential_training_results(J, losses_train, losses_test, config)

    # Compute extrapolation evaluation on both train and test sets
    extrapolation_evaluation_gradient_flow(config, J, t_train)
