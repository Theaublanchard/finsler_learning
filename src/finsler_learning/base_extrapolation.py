from dataclasses import dataclass
from datetime import datetime
import torch
from torch import Tensor
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import CubicSpline
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
    if torch.isnan(traj).any():
        nan_batches = traj.isnan().any(dim=(1, 2))
        traj = traj[~nan_batches]
        logger.warning(
            f"Removed {nan_batches.sum().item()} trajectories containing NaN values."
        )
    if traj.shape[0] == 0:
        return None
    return traj


def evaluate_dataloader(
    exp_map: RandomExpMapRiemann | RandomExpMapRanders,
    dataloader: torch.utils.data.DataLoader,
    config: ExtrapolationEvalRiemannConfig,
    n_run: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Evaluate the dataloader by extrapolating trajectories and filtering them.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader to evaluate.
    config : ExtrapolationEvalRiemannConfig
        The configuration object containing all necessary parameters and models.

    Returns
    -------
    all_trajs : torch.Tensor (N_samples,T,d)
        The extrapolated trajectories for all samples in the dataloader.
    res : dict[str, torch.Tensor]
        The computed monotonicity measures for each metric in ALL_METRICS.
    t_pred : torch.Tensor (N_samples,T)
        The predicted time for each point in the extrapolated trajectories.
    """
    res: dict[str : list[torch.Tensor]] = {name: [] for name in ALL_METRICS.keys()}
    all_trajs = []
    all_t_pred = []

    pbar = tqdm(
        enumerate(dataloader),
        desc=f"Extrapolation evaluation - Run {n_run}",
        total=len(dataloader),
        unit="batch",
    )
    for i, data in pbar:
        x, t = data
        x = x.to(config.device)
        t = t.to(config.device)

        z, _ = config.encoder(x)
        traj = exp_map(z)[0].detach()  # (B,T,d)
        traj = filter_traj_nan(traj)

        if traj is None:
            continue

        all_trajs.append(traj.cpu())

        t_pred = retrieve_time_from_traj(
            traj,
            decoder=config.decoder,
            regressor=config.regressor,
            use_pbar=False,
        )  # (B,T)

        if config.post_process_labels is not None:
            t_pred = config.post_process_labels(t_pred)
        all_t_pred.append(t_pred.cpu())

        for name, metric_fn in ALL_METRICS.items():
            mono_measure = metric_fn(t_pred)  # (B,)
            res[name].append(mono_measure.cpu())

    all_trajs = torch.cat(all_trajs, dim=0)  # (N_samples,T,d)
    all_t_pred = torch.cat(all_t_pred, dim=0)  # (N_samples,T)

    for name in ALL_METRICS.keys():
        res[name] = torch.cat(res[name], dim=0)  # (N_samples,)

    return all_trajs, res, all_t_pred


def compute_extrapolation_and_monotonicity(
    config: ExtrapolationEvalRiemannConfig,
    dataloader_test: torch.utils.data.DataLoader,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Compute the extrapolated trajectories and monotonicity measures for the test set.

    Parameters
    ----------
    config : ExtrapolationEvalRiemannConfig
        Configuration object containing all necessary parameters and models.
    dataloader_test : torch.utils.data.DataLoader
        The dataloader for the test set, used to compute the extrapolated trajectories and monotonicity measures.
    encoder : torch.nn.Module
        The encoder used to compute the latent representation.

    Returns
    -------
    all_trajs : torch.Tensor (N_pts*n_runs,T,d)
        The extrapolated trajectories for all samples in the test set.
    res : dict[str, torch.Tensor] (N_pts*n_runs,)
        The computed monotonicity measures for each metric in ALL_METRICS.
    t_pred : torch.Tensor (N_pts*n_runs,T)
        The predicted time for each point in the extrapolated trajectories.
    """
    exp_map = get_random_exp_map(config)

    all_trajs = []
    all_preds = []
    res = {name: [] for name in ALL_METRICS.keys()}

    for run in range(config.n_runs):

        all_trajs_, res_, t_pred_ = evaluate_dataloader(
            exp_map=exp_map,
            dataloader=dataloader_test,
            config=config,
            n_run=run,
        )

        all_trajs.extend(all_trajs_)
        all_preds.extend(t_pred_)
        for name in ALL_METRICS.keys():
            res[name].extend(res_[name])

    t_pred = torch.stack(all_preds, dim=0)  # (N_pts*n_runs,T)
    all_trajs = torch.stack(all_trajs, dim=0)  # (N_pts*n_runs,T,d)
    for name in ALL_METRICS.keys():
        res[name] = torch.stack(res[name], dim=0)  # (N_pts*n_runs,)

    return all_trajs, res, t_pred


def export_extrapolation_evaluation(
    config: ExtrapolationEvalRiemannConfig,
    metrics_test: dict[str, Tensor],
) -> None:
    """
    Exports the geodesic evaluation results to a file.

    Parameters:
    ----------
    config : ExtrapolationEvalRiemannConfig
        Configuration object containing export path.
    mono_measure_list_test : torch.Tensor (N_test,N_pts)
        The computed monotonicity measures for the test set.

    """
    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    for name in ALL_METRICS.keys():
        torch.save(metrics_test[name], export_path / f"extrapolation_{name}_test.pth")


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
    """Perform extrapolation evaluation using the provided configuration.

    Parameters
    ----------
    config : ExtrapolationEvalRiemannConfig
        Configuration object containing all necessary parameters and models.

    Returns
    -------
    traj_test : torch.Tensor (N_pts*n_runs,T,d)
        The extrapolated trajectories for all samples in the test set.
    t_pred_test : torch.Tensor (N_pts*n_runs,T)
        The predicted time for each point in the extrapolated trajectories for the test set.
    """
    start_time = datetime.now()

    logger.info("Computing extrapolation and monotonicity for test set...")
    traj_test, res_test, t_pred_test = compute_extrapolation_and_monotonicity(
        config,
        config.dataloader_test,
    )

    summary_test = get_metrics_summary(res_test)
    logger.info("Extrapolation evaluation results:")
    for name in ALL_METRICS.keys():
        logger.info(
            f"{name.upper()}"
            f" - Test: mean={summary_test[name]['mean']:.4f} +- {summary_test[name]['std']:.4f}, "
        )

    logger.info("Exporting results...")
    export_extrapolation_evaluation(
        config,
        res_test,
    )
    export_extrapolated_trajectories(
        config,
        traj_test,
    )
    export_extrapolated_time(
        config,
        t_pred_test,
    )

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info(
        f"Extrapolation evaluation completed in {elapsed_time} for {config.export_path}"
    )
    return traj_test, t_pred_test
