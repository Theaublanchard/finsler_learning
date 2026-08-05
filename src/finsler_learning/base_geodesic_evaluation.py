from geodesic_toolbox import (
    CoMetric,
    SolverGraphGEORCE,
    SolverGraphGEORCEFinsler,
    RandersMetrics,
    SolverGraphSpline,
    SolverGraphSplineFinsler,
)
from dataclasses import dataclass
import torch
from torch import Tensor
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
import logging
from datetime import datetime

from .lgds import LGDS
from .models import BaseEncoder, BaseDecoder
from .utils import kendall_tau, pearson_corr, spearman_corr

LOGGER = logging.getLogger(__name__)


ALL_METRICS = {
    "kendall_tau": kendall_tau,
    "pearson_corr": pearson_corr,
    "spearman_corr": spearman_corr,
}


@dataclass
class GeodesicSolverParams:
    n_pts: int = 100
    n_neighbors: int = 8
    max_iter_georce: int = 100
    max_iter_spline: int = 500
    lr_spline: float = 1e-2
    tol_solver: float = 1e-3
    needs_spline: bool = False
    batch_size: int = 32
    max_data_count: int = 10_000
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pbar: bool = True  # whether to show a progress bar during geodesic computation

@dataclass
class GeodesicEvalRiemannConfig:
    export_path: Path
    dataloader_train: torch.utils.data.DataLoader
    dataloader_test: torch.utils.data.DataLoader
    encoder: BaseEncoder
    decoder: BaseDecoder
    cometric: CoMetric
    solver_params: GeodesicSolverParams
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_regressor: torch.nn.Module | None = None
    regressor: torch.nn.Module | None = None

    # n_run_dataset: int = 1
    post_process_label: callable = None
    latent_classifier: torch.nn.Module | None = None
    export_geodesics: bool = True
    # used for quality evaluation. Represents the images to use as reference for the quality evaluation of the generated geodesics.
    reference_dataloader_train: torch.utils.data.DataLoader | None = None
    export_then_evaluate: bool = False  # whether to export the geodesics then evaluate them
    # This function is used when export_then_evaluate is True
    # Signature of the function:
    # def custom_export(decoded_images: Tensor (B, T, ...),
    #                   config: GeodesicEvalRiemannConfig,
    #                   train:bool  ) -> None:
    # In which case the regressor must be able to process the decoded images
    # from the saved trajectories
    custom_export: callable = None  # custom function to export geodesics


@dataclass
class GeodesicEvalFinslerConfig(GeodesicEvalRiemannConfig):
    randers_metric: RandersMetrics = None


@torch.no_grad()
def embed_dataset(
    dataloader: torch.utils.data.DataLoader,
    encoder: BaseEncoder,
    device: torch.device,
) -> Tensor:
    """
    Embeds the dataset using the provided encoder.

    Parameters:
    -----------
    dataloader : torch.utils.data.DataLoader
        The dataloader containing the dataset.
    encoder : BaseEncoder
        The encoder model to use for embedding.
    device : torch.device
        The device to run the computations on.

    Returns:
    --------
    embeds : torch.Tensor (N, D)
        The embedded representations of the dataset.
    """
    encoder.eval()
    embeddings = []
    for data in dataloader:
        x, _ = data
        x = x.to(device)
        z, _ = encoder(x)
        embeddings.append(z)
    embeds = torch.cat(embeddings, dim=0)
    return embeds


@torch.no_grad()
def embed_dataset_w_labels(
    dataloader: torch.utils.data.DataLoader,
    encoder: BaseEncoder,
    device: torch.device,
) -> Tensor:
    """
    Embeds the dataset using the provided encoder.

    Parameters:
    -----------
    dataloader : torch.utils.data.DataLoader
        The dataloader containing the dataset.
    encoder : torch.nn.Module
        The encoder model to use for embedding.
    device : torch.device
        The device to run the computations on.

    Returns:
    --------
    embeds : torch.Tensor (N, D)
        The embedded representations of the dataset.
    labels : torch.Tensor (N,)
        The labels corresponding to the embedded representations.
    """
    encoder.eval()
    embeddings = []
    labels = []
    for data in dataloader:
        x, l = data
        x = x.to(device)
        z, _ = encoder(x)
        embeddings.append(z)
        labels.append(l)
    embeds = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeds, labels


def get_solver(
    config: GeodesicSolverParams,
    cometric: CoMetric,
    dataloader_train: torch.utils.data.DataLoader,
    encoder: BaseEncoder,
    device: torch.device,
    randers_metric: RandersMetrics = None,
    is_lgds: bool = False,
) -> SolverGraphGEORCE | SolverGraphGEORCEFinsler:
    if is_lgds:
        embeds, labels = embed_dataset_w_labels(dataloader_train, encoder, device)
        lgds = LGDS(T=config.n_pts, d=embeds.shape[1])
        lgds.fit(embeds.cpu(), labels.cpu())
        return lgds.to(device)

    embeds = embed_dataset(dataloader_train, encoder, device)
    if randers_metric is not None:
        if not config.needs_spline:
            solver = SolverGraphGEORCEFinsler(
                finsler=randers_metric,
                data=embeds,
                n_neighbors=config.n_neighbors,
                max_iter=config.max_iter_georce,
                tol=config.tol_solver,
                batch_size=config.batch_size,
                max_data_count=config.max_data_count,
                T=config.n_pts,
                smooth_curve=False,
            )
        else:
            solver = SolverGraphSplineFinsler(
                finsler=randers_metric,
                data=embeds,
                n_neighbors=config.n_neighbors,
                batch_size=config.batch_size,
                max_data_count=config.max_data_count,
                T=config.n_pts,
                lr=config.lr_spline,
                n_iter=config.max_iter_spline,
            )
    else:
        if not config.needs_spline:
            solver = SolverGraphGEORCE(
                cometric=cometric,
                data=embeds,
                n_neighbors=config.n_neighbors,
                max_iter=config.max_iter_georce,
                tol=config.tol_solver,
                batch_size=config.batch_size,
                max_data_count=config.max_data_count,
                T=config.n_pts,
                smooth_curve=False,
            )
        else:
            solver = SolverGraphSpline(
                cometric=cometric,
                data=embeds,
                n_neighbors=config.n_neighbors,
                batch_size=config.batch_size,
                max_data_count=config.max_data_count,
                T=config.n_pts,
                lr=config.lr_spline,
                n_iter=config.max_iter_spline,
            )
    solver.to(config.device)
    return solver


def get_ordered_data_random(x: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Given a batch of data and their corresponding time points,
    returns a random splitting of the data such that (t_0 <= t_1).all() is True.

    Parameters:
    -----------
    x : torch.Tensor (B,...)
        The input data tensor.
    t : torch.Tensor (B,)
        The time points corresponding to the data.

    Returns:
    --------
    x_0 : torch.Tensor (B//2,...)
        The first half of the input data tensor.
    t_0 : torch.Tensor (B//2,)
        The time points corresponding to the first half of the data.
    x_1 : torch.Tensor (B//2,...)
        The second half of the input data tensor.
    t_1 : torch.Tensor (B//2,)
        The time points corresponding to the second half of the data.
    """
    B = x.shape[0]
    assert B % 2 == 0, "Batch size must be even for splitting."
    perm = torch.randperm(B)
    idx_0 = perm[: B // 2]
    idx_1 = perm[B // 2 : B]

    x_0 = x[idx_0]
    t_0 = t[idx_0]
    x_1 = x[idx_1]
    t_1 = t[idx_1]

    # Ensure constrains, else swap
    mask = t_0 > t_1
    x_0_temp = x_0.clone()
    x_0[mask] = x_1[mask]
    x_1[mask] = x_0_temp[mask]

    t_0_temp = t_0.clone()
    t_0[mask] = t_1[mask]
    t_1[mask] = t_0_temp[mask]

    return x_0, t_0, x_1, t_1


def get_ordered_data_fixed(
    x: Tensor, t: Tensor, return_idx: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Given a batch of data and their corresponding time points,
    returns a fixed splitting of the data such that (t_0 <= t_1).all() is True.
    The split ensures maximum distances between t_0 and t_1.

    Parameters:
    -----------
    x : torch.Tensor (B,...)
        The input data tensor.
    t : torch.Tensor (B,)
        The time points corresponding to the data.

    Returns:
    --------
    x_0 : torch.Tensor (B//2,...)
        The first half of the input data tensor.
    t_0 : torch.Tensor (B//2,)
        The time points corresponding to the first half of the data.
    x_1 : torch.Tensor (B//2,...)
        The second half of the input data tensor.
    t_1 : torch.Tensor (B//2,)
        The time points corresponding to the second half of the data.
    if return_idx:
        idx_0 : torch.Tensor (B//2,)
            The indices of the first half of the data.
        idx_1 : torch.Tensor (B//2,)
            The indices of the second half of the data.
    """
    B = x.shape[0]
    idx = torch.argsort(t)
    # If even all good,
    if B % 2 == 0:
        idx_0 = idx[: B // 2]
        idx_1 = idx[B // 2 :]
    # Else, reject the middle point and split the rest
    else:
        logger = logging.getLogger(__name__)
        logger.warning(f"Batch size {B} is odd, rejecting the middle point for splitting.")
        idx_0 = idx[: B // 2]
        idx_1 = idx[B // 2 + 1 :]

    x_0 = x[idx_0]
    t_0 = t[idx_0]
    x_1 = x[idx_1]
    t_1 = t[idx_1]
    if return_idx:
        return x_0, t_0, x_1, t_1, idx_0, idx_1
    return x_0, t_0, x_1, t_1


@torch.no_grad()
def retrieve_time_from_traj(
    traj: Tensor,
    latent_regressor=None,
    decoder: BaseDecoder = None,
    regressor=None,
    batch_size: int = 128,
    use_pbar: bool = True,
) -> Tensor:
    """
    Given a trajectory tensor, retrieves the time points from the last dimension.

    Parameters:
    -----------
    traj : torch.Tensor (B, T, D)
        The latent trajectory tensor where T is the number of time points.
    latent_regressor : torch.nn.Module, optional
        A model that directly regresses time from the latent space. If provided, it will be used instead of the decoder and regressor.
    decoder : BaseDecoder, optional
        The decoder model to use for decoding the latent trajectories into image space. Required if latent_regressor is not provided.
    regressor : torch.nn.Module, optional
        The regressor model to use for regressing time from the decoded images. Required if latent_regressor is not provided.
    batch_size : int
        The batch size to use when processing the trajectories. This is used to avoid memory issues when decoding large trajectories.

    Returns:
    --------
    t_pred : torch.Tensor (B,T)
        The predicted time points corresponding to the trajectories.
    """
    B, T, D = traj.shape
    traj = rearrange(traj, "b t d -> (b t) d")
    dtype = next(decoder.parameters()).dtype
    device = next(decoder.parameters()).device
    regressor = regressor.to(dtype)
    assert latent_regressor is not None or (
        decoder is not None and regressor is not None
    ), "Either latent classifier or both decoder and regressor must be provided to infer timestamps."
    t_pred_list = []
    pbar = tqdm(
        range(0, traj.shape[0], batch_size),
        desc="Retrieving time from trajectory",
        total=(traj.shape[0] + batch_size - 1) // batch_size,
        unit="batch",
        leave=use_pbar,
    )
    if latent_regressor is not None:
        for i in pbar:
            end = min(i + batch_size, traj.shape[0])
            traj_batch = traj[i:end]  # (batch_size, D)
            t_pred_ = latent_regressor(
                traj_batch.to(dtype).to(device)
            ).cpu()  # ( batch_size, 1)
            t_pred_list.append(t_pred_)
    else:
        for i in pbar:
            end = min(i + batch_size, traj.shape[0])
            traj_batch = traj[i:end]  # (batch_size, D)
            img = decoder(traj_batch.to(dtype).to(device))  # (batch_size, ...)
            y_hat = regressor(img).cpu()
            t_pred_list.append(y_hat)
    t_pred = torch.cat(t_pred_list, dim=0)  # (B*T, 1)
    t_pred = rearrange(t_pred, "(b t) -> b t", b=B)  # (B, T)
    return t_pred.to(traj.device)


def export_latent_geodesics(
    config: GeodesicEvalRiemannConfig | GeodesicEvalFinslerConfig,
    dataloader: torch.utils.data.DataLoader,
    solver: torch.nn.Module,
    train: bool,
) -> None:
    """
    Computes and exports the latent geodesics for a given dataloader.

    Parameters:
    ----------
    config : GeodesicEvalRiemannConfig or GeodesicEvalFinslerConfig
        Configuration object containing export path and other parameters.
    dataloader : torch.utils.data.DataLoader
        The dataloader for which to compute the geodesics.
    solver : torch.nn.Module
        The solver module to use for computing the geodesics.
    train : bool
        Whether to compute geodesics for the training set.
    """
    pbar = tqdm(
        enumerate(dataloader),
        desc="Exporting latent geodesics",
        total=len(dataloader),
        unit="batch",
    )
    traj_list = []
    dst_list = []
    for i, data in pbar:
        x, t = data
        x = x.to(config.device)
        t = t.to(config.device)
        z, _ = config.encoder(x)

        z_0, t_0, z_1, t_1 = get_ordered_data_fixed(z, t)
        traj = solver.get_trajectories(z_0, z_1)  #  (B, T, latent_dim)
        dst = solver.compute_distance(traj)  # (B,)
        traj_list.append(traj.cpu().detach())
        dst_list.append(dst.cpu().detach())

    traj_list = torch.cat(traj_list, dim=0)  # (N, T, latent_dim)
    dst_list = torch.cat(dst_list, dim=0)  # (N,)
    name_t = f"latent_geodesics_{'train' if train else 'test'}.pth"
    name_dst = f"geodesic_distance_{'train' if train else 'test'}.pth"
    config.export_path.mkdir(parents=True, exist_ok=True)
    torch.save(traj_list, config.export_path / name_t)
    torch.save(dst_list, config.export_path / name_dst)


def compute_geodesic_stats(
    config: GeodesicEvalRiemannConfig,
) -> tuple[Tensor, dict[str, Tensor], Tensor, dict[str, Tensor]]:
    """
    Evaluates the geodesic distance and monotonicity measure for a given dataset.

    Parameters:
    ----------
    config : GeodesicEvalRiemannConfig
        Configuration object

    Returns:
    -------
    dst_list_test : torch.Tensor (N,)
        The computed geodesic distances for the test set
    metrics_list_test : dict of torch.Tensor (N,)
        The computed additional metrics for the test set
    """
    solver = get_solver(
        config=config.solver_params,
        cometric=config.cometric,
        dataloader_train=config.dataloader_train,
        encoder=config.encoder,
        device=config.device,
        randers_metric=getattr(config, "randers_metric", None),
        is_lgds=config.export_path.name == "geodesic_evaluation_lgds",
    )

    if config.export_geodesics:
        LOGGER.info("Exporting latent geodesics for test set...")
        export_latent_geodesics(config, config.dataloader_test, solver, train=False)

    latent_traj_test = torch.load(config.export_path / "latent_geodesics_test.pth")

    LOGGER.info(f"Shape of the test latent trajectories: {latent_traj_test.shape}")

    dst_list_test = torch.load(config.export_path / "geodesic_distance_test.pth")

    LOGGER.info("Retrieving time from trajectories")
    t_pred_test = retrieve_time_from_traj(
        traj=latent_traj_test,
        latent_regressor=config.latent_regressor,
        decoder=config.decoder,
        regressor=config.regressor,
    )  # (B,T)

    if config.post_process_label is not None:
        LOGGER.info("Post processing predicted time")
        # Ensure everything on the correct device
        config.latent_classifier = config.latent_classifier.to(config.device)
        t_pred_test = t_pred_test.to(config.device)
        latent_traj_test = latent_traj_test.to(config.device)
        t_pred_test = config.post_process_label(
            latent_traj_test, t_pred_test, config.latent_classifier
        )
    # Export the time
    torch.save(t_pred_test, config.export_path / "geo_t_pred_test.pth")

    LOGGER.info("Computing metrics from retrieved time")
    metrics_list_test = {name: [] for name in ALL_METRICS}
    mean_metrics_test = {name: [] for name in ALL_METRICS}
    for name, metric_fn in ALL_METRICS.items():
        m_test = metric_fn(t_pred_test).cpu().detach()
        metrics_list_test[name].append(m_test)
        mean_metrics_test[name].append(m_test.mean().item())

    LOGGER.info(f"Geodesic evaluation results:\n  Test set:{mean_metrics_test}")
    return dst_list_test, metrics_list_test


@torch.no_grad()
def decode_trajectories(traj: Tensor, decoder: BaseDecoder) -> Tensor:
    """
    Decodes a batch of latent trajectories into image space.
    To save memory, we either loop over time points or batch points depending on their sizes.

    Parameters:
    -----------
    traj : torch.Tensor (B, T, D)
        The latent trajectory tensor where T is the number of time points.
    decoder : BaseDecoder
        The decoder model to use for decoding.

    Returns:
    --------
    decoded_imgs : torch.Tensor (B, T, ...)
        The decoded images corresponding to the latent trajectories.
    """
    # Parallel computing different if B >= T or T > B
    B, T, D = traj.shape
    if T > B:
        traj = rearrange(traj, "b t d -> t b d")  # (T, B, D)
    decoded_img_list = []
    for idx in range(traj.shape[0]):
        z_t = traj[idx]
        x_recon_t = decoder(z_t)
        decoded_img_list.append(x_recon_t)
    decoded_imgs = torch.stack(decoded_img_list, dim=0)

    if T > B:
        decoded_imgs = rearrange(decoded_imgs, "t b ... -> b t ...")
    return decoded_imgs


def export_geodesic_evaluation(
    config: GeodesicEvalRiemannConfig,
    dst_list_test: Tensor,
    metrics_list_test: dict[str, Tensor],
) -> None:
    """
    Exports the geodesic evaluation results to a file.

    Parameters:
    ----------
    config : GeodesicEvalRiemannConfig
        Configuration object containing export path.
    dst_list_test : torch.Tensor (N,)
        The computed geodesic distances for the test set.
    metrics_list_test : dict[str, torch.Tensor] (N,)
        The computed monotonicity measures for the test set.
    """
    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    torch.save(dst_list_test, export_path / "geo_dst_list_test.pth")
    for name, metric_values in metrics_list_test.items():
        torch.save(metric_values, export_path / f"geo_{name}_list_test.pth")


def geodesic_evaluation(
    config: GeodesicEvalRiemannConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Main function to perform geodesic evaluation.

    Parameters:
    ----------
    config : GeodesicEvalRiemannConfig
        Configuration object containing all necessary parameters.

    Returns:
    -------
    dst_list_test : torch.Tensor (N,)
        The computed geodesic distances for the test set.
    metrics_list_test : dict[str, torch.Tensor] (N,)
        The computed monotonicity measures for the test set.
    """
    start_time = datetime.now()

    dst_test, metrics_list_test = compute_geodesic_stats(config)
    export_geodesic_evaluation(config, dst_test, metrics_list_test)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    LOGGER.info(
        f"Geodesic evaluation completed in {elapsed_time} for config {config.export_path}"
    )
    return dst_test, metrics_list_test
