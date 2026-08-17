from dataclasses import dataclass
from datetime import datetime
import torch
from torch import Tensor
from pathlib import Path
from tqdm import tqdm
import logging

from geodesic_toolbox import (
    CoMetric,
    RandersMetrics,
    ExpMapRiemann,
    ExpMapRanders,
)
from .base_geodesic_evaluation import retrieve_time_from_traj, ALL_METRICS
from .base_random_exp_map import RandomExpMapRiemann, RandomExpMapRanders

logger = logging.getLogger(__name__)


@dataclass
class ExtrapolationEvalRiemannConfig:
    latent_dim: int
    export_path: Path
    dataloader_train: torch.utils.data.DataLoader
    dataloader_test: torch.utils.data.DataLoader
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    cometric: CoMetric
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regressor: torch.nn.Module | None = None
    batch_size: int = 8
    T: int = 100
    T_max: float = 1.0
    n_runs: int = 1
    alpha: float = 1.0
    substeps: int = 1
    solver_type: str = "leapfrog_implicit"
    post_process_labels: callable = None
    reference_dataloader_train: torch.utils.data.DataLoader | None = None
    moving_avg_momentum: float | None = 0.8
    # if not None, the momentum are updated with a moving average with this momentum instead of being resampled at each step


@dataclass
class ExtrapolationEvalFinslerConfig(ExtrapolationEvalRiemannConfig):
    randers_metric: RandersMetrics = None
    eta: float = 0.9  # target expected cosine similarity for the random shooting direction


def get_kappa(d: int, eta: float) -> float:
    """
    Get the kappa value for the von Mises-Fisher distribution to achieve
    a target expexted cosine similarity of samples and mean direction of eta.
    We use the approximation from Eq. 4.4 of
    Clustering on the Unit Hypersphere using von Mises-Fisher Distributions

    Parameters:
    d : int
        The dimension of the space.
    eta : float
        The target expected cosine similarity between the samples and the mean direction.

    Returns:
    kappa : float
        The concentration parameter
    """
    return (eta * d - eta**3) / (1 - eta**2)


def get_exp_map(
    config: ExtrapolationEvalRiemannConfig,
) -> ExpMapRiemann | ExpMapRanders:
    """Get the exponential map object based on the configuration."""
    if isinstance(config, ExtrapolationEvalFinslerConfig):
        exp_map = ExpMapRanders(
            randers=config.randers_metric,
            T_max=config.T_max,
            T=config.T,
            solver=config.solver_type,
            substeps=config.substeps,
        )
    else:
        exp_map = ExpMapRiemann(
            cometric=config.cometric,
            T=config.T,
            T_max=config.T_max,
            solver=config.solver_type,
            substeps=config.substeps,
        )
    return exp_map


def get_random_exp_map(
    config: ExtrapolationEvalRiemannConfig,
) -> RandomExpMapRiemann | RandomExpMapRanders:
    """Get the exponential map object based on the configuration."""
    if isinstance(config, ExtrapolationEvalFinslerConfig):
        exp_map = RandomExpMapRanders(
            randers=config.randers_metric,
            T_max=config.T_max,
            T=config.T,
            solver=config.solver_type,
            substeps=config.substeps,
            alpha=config.alpha,
            kappa=get_kappa(d=config.latent_dim, eta=config.eta),
            moving_avg_momentum=config.moving_avg_momentum,
        )
    else:
        exp_map = RandomExpMapRiemann(
            cometric=config.cometric,
            T=config.T,
            T_max=config.T_max,
            solver=config.solver_type,
            substeps=config.substeps,
            alpha=config.alpha,
            moving_avg_momentum=config.moving_avg_momentum,
        )
    return exp_map


def filter_traj_nan(traj: torch.Tensor) -> torch.Tensor | None:
    """
    Remove the nan trajectories containing NaN values.
    If only NaNs, return None.

    Parameters
    ----------
    traj : torch.Tensor (B,T,d)
        The trajectory to filter.

    Returns
    -------
    torch.Tensor (B',T,d) | None
        The filtered trajectory or None if all values are NaN.
    """
    B, T, d = traj.shape
    if torch.isnan(traj).any():
        nan_batches = traj.isnan().any(dim=(1, 2))
        traj = traj[~nan_batches]
        logger.warning(
            f"Removed {nan_batches.sum().item()}/{B} trajectories containing NaN values."
        )
    if traj.shape[0] == 0:
        return None
    return traj


def compute_extrapolated_trajectories(
    config: ExtrapolationEvalRiemannConfig,
    dataloader_test: torch.utils.data.DataLoader,
) -> Tensor:
    """
    Compute the extrapolated trajectories for the test set.

    Parameters
    ----------
    config : ExtrapolationEvalRiemannConfig
        Configuration object containing the necessary parameters for extrapolation.
    dataloader_test : torch.utils.data.DataLoader
        DataLoader for the test set.

    Returns
    -------
    torch.Tensor (N_test * n_runs, T, latent_dim)
        The extrapolated trajectories for the test set.
    """
    exp_map = get_random_exp_map(config)
    all_trajs = []

    for run in range(config.n_runs):
        pbar = tqdm(
            enumerate(dataloader_test),
            desc=f"Extrapolation - Run {run}",
            total=len(dataloader_test),
            unit="batch",
        )

        for _, data in pbar:
            x, _ = data
            x = x.to(config.device)

            z, _ = config.encoder(x)
            traj = exp_map(z)[0].detach()  # (B,T,d)
            traj = filter_traj_nan(traj)

            if traj is not None:
                all_trajs.append(traj.cpu())

    if not all_trajs:
        raise RuntimeError("No valid extrapolated trajectories were generated.")

    # (N_test * n_runs, T, latent_dim)
    return torch.cat(all_trajs, dim=0)


@torch.no_grad()
def evaluate_extrapolated_trajectories(
    config: ExtrapolationEvalRiemannConfig,
    traj_test: Tensor,
) -> tuple[dict[str, Tensor], Tensor]:
    """
    Evaluate already-computed extrapolated trajectories.
    
    Parameters
    ----------
    config : ExtrapolationEvalRiemannConfig
        Configuration object containing the necessary parameters for evaluation.
    traj_test : torch.Tensor (N_test * n_runs, T, latent_dim)
        The extrapolated trajectories for the test set.

    Returns
    -------
    metrics_test : dict[str, torch.Tensor]
        A dictionary containing the computed evaluation metrics for the test set.
    t_pred_test : torch.Tensor (N_test * n_runs, T)
        The predicted time for each point in the extrapolated trajectories for the test set.
    """
    logger.info("Retrieving time from extrapolated trajectories...")
    t_pred_test = retrieve_time_from_traj(
        traj=traj_test,
        decoder=config.decoder,
        regressor=config.regressor,
    )

    if config.post_process_labels is not None:
        t_pred_test = config.post_process_labels(t_pred_test)

    metrics_test = {
        name: metric_fn(t_pred_test).cpu().detach() for name, metric_fn in ALL_METRICS.items()
    }

    return metrics_test, t_pred_test.cpu().detach()


def export_extrapolated_trajectories(
    config: ExtrapolationEvalRiemannConfig,
    traj_test: Tensor,
) -> None:
    """Export the extrapolated trajectories to a file."""
    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)
    torch.save(traj_test, export_path / "extrapolated_traj_test.pth")


def compute_extrapolation_evaluation(
    config: ExtrapolationEvalRiemannConfig,
) -> tuple[dict[str, Tensor], Tensor]:
    """Evaluate trajectories previously exported to ``config.export_path``."""
    traj_test = torch.load(config.export_path / "extrapolated_traj_test.pth")
    logger.info(f"Shape of the test extrapolated trajectories: {traj_test.shape}")

    metrics_test, t_pred_test = evaluate_extrapolated_trajectories(
        config=config,
        traj_test=traj_test,
    )

    torch.save(
        t_pred_test,
        config.export_path / "extrapolated_time_test.pth",
    )
    return metrics_test, t_pred_test


def export_extrapolation_evaluation(
    config: ExtrapolationEvalRiemannConfig,
    metrics_test: dict[str, Tensor],
) -> None:
    """Export extrapolation evaluation metrics to files."""
    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    for name, metric_values in metrics_test.items():
        torch.save(
            metric_values,
            export_path / f"extrapolation_{name}_test.pth",
        )


def get_metrics_summary(metrics: dict[str, Tensor]) -> dict[str, float]:
    """Compute summary statistics for the given metrics.

    Parameters
    ----------
    metrics : dict[str, torch.Tensor]
        A dictionary containing the computed monotonicity measures for each metric.

    Returns
    -------
    dict[str, float]
        A dictionary containing the mean and standard deviation for each metric.
    """
    summary = {}
    for name, values in metrics.items():
        mean = values.mean().item()
        std = values.std().item()
        summary[name] = {"mean": mean, "std": std}
    return summary


def export_extrapolated_trajectories(
    config: ExtrapolationEvalRiemannConfig,
    traj_test: Tensor,
) -> None:
    """
    Exports the extrapolated trajectories to a file.

    Parameters:
    ----------
    config : ExtrapolationEvalRiemannConfig
        Configuration object containing export path.
    traj_test : torch.Tensor (N_test*N_pts,T,d)
        The extrapolated trajectories for the test set.
    """
    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    torch.save(traj_test, export_path / f"extrapolated_traj_test.pth")


def export_extrapolated_time(
    config: ExtrapolationEvalRiemannConfig,
    t_pred_test: Tensor,
) -> None:
    """
    Exports the predicted time for the extrapolated trajectories to a file.

    Parameters:
    ----------
    config : ExtrapolationEvalRiemannConfig
        Configuration object containing export path.
    t_pred_test : torch.Tensor (N_test*N_pts,T)
        The predicted time for each point in the extrapolated trajectories for the test set.
    """
    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    torch.save(t_pred_test, export_path / f"extrapolated_time_test.pth")


def extrapolation_evaluation(
    config: ExtrapolationEvalRiemannConfig,
) -> tuple[Tensor, Tensor]:
    """Compute/export trajectories first, then evaluate the exported trajectories."""
    start_time = datetime.now()

    # Phase 1: compute and export trajectories.
    logger.info("Computing extrapolated trajectories for test set...")
    traj_test = compute_extrapolated_trajectories(
        config=config,
        dataloader_test=config.dataloader_test,
    )
    export_extrapolated_trajectories(config, traj_test)

    # Phase 2: load the exported trajectories and evaluate them.
    logger.info("Evaluating exported extrapolated trajectories...")
    metrics_test, t_pred_test = compute_extrapolation_evaluation(config)

    summary_test = get_metrics_summary(metrics_test)
    logger.info("Extrapolation evaluation results:")
    for name in ALL_METRICS.keys():
        logger.info(
            f"{name.upper()}"
            f" - Test: mean={summary_test[name]['mean']:.4f} +- "
            f"{summary_test[name]['std']:.4f}, "
        )

    export_extrapolation_evaluation(config, metrics_test)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info(
        f"Extrapolation evaluation completed in {elapsed_time} for {config.export_path}"
    )
    return traj_test, t_pred_test
