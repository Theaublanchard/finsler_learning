import torch
import logging
from torch import nn, Tensor
from torch.utils.data import DataLoader

from einops import rearrange
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torcheval.metrics.image.fid import FIDInceptionV3

from pytorchfwd.fwd import compute_packet_statistics, _compute_avg_frechet_distance

from .base_geodesic_evaluation import (
    GeodesicEvalRiemannConfig,
    get_solver,
    export_latent_geodesics,
)
from .base_extrapolation import ExtrapolationEvalRiemannConfig

LOGGER = logging.getLogger(__name__)


class MetricEvaluator(nn.Module):
    def __init__(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_runs: int = 2,
    ):
        super(MetricEvaluator, self).__init__()
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.device = device
        self.n_runs = n_runs

    def forward(self):
        raise NotImplementedError("Subclasses should implement this method.")


class CLIPMetricEvaluator(MetricEvaluator):
    """
    Compute CLIP Score between images and captions.
    Formally for images I_train, I_test we compute
     E[ CLIP(I_train, I_test) ]
    """

    def __init__(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_runs: int = 2,
    ):
        super(CLIPMetricEvaluator, self).__init__(
            dataloader_train, dataloader_test, device=device, n_runs=n_runs
        )
        self.clip_metric = CLIPScore().to(self.device)
        assert (
            dataloader_test.batch_size == dataloader_train.batch_size
        ), f"Dataloader batch sizes must be equal for CLIP metric evaluation.Got train: {dataloader_train.batch_size}, test: {dataloader_test.batch_size}"

    def pre_process_images(self, x_train: Tensor, x_test: Tensor) -> tuple[Tensor, Tensor]:
        # Clip score requires RGB images
        if x_train.shape[1] == 1:
            x_train = x_train.repeat(1, 3, 1, 1)
            x_test = x_test.repeat(1, 3, 1, 1)

        # if different batch sizes, pad the min with last image
        b_train, b_test = x_train.shape[0], x_test.shape[0]
        if b_train < b_test:
            pad_size = b_test - b_train
            pad_images = x_train[-1:].repeat(pad_size, 1, 1, 1)
            x_train = torch.cat([x_train, pad_images], dim=0)
        elif b_test < b_train:
            pad_size = b_train - b_test
            pad_images = x_test[-1:].repeat(pad_size, 1, 1, 1)
            x_test = torch.cat([x_test, pad_images], dim=0)
        return x_train, x_test

    def forward(self):
        self.clip_metric.reset()
        n = min(len(self.dataloader_train), len(self.dataloader_test))
        pbar = tqdm(total=n * self.n_runs, desc="Computing CLIP Score", unit="batches")
        for _ in range(self.n_runs):
            for i_train, i_test in zip(self.dataloader_train, self.dataloader_test):
                x_train, *_ = i_train
                x_test, *_ = i_test
                x_train = x_train.to(self.device)
                x_test = x_test.to(self.device)

                x_train, x_test = self.pre_process_images(x_train, x_test)

                self.clip_metric.update(x_train, x_test)

                pbar.update(1)
                pbar.set_postfix({"CLIP Score": self.clip_metric.compute().item()})
        pbar.close()

        return self.clip_metric.compute().item()


class KIDMetricEvaluator(MetricEvaluator):
    """
    Compute Kernel Inception Distance (KID) between images.

    Parameters
    ----------
    dataloader_train : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    dataloader_test : torch.utils.data.DataLoader
        DataLoader for the testing dataset.
    normalize : bool, optional
        Set to True if the images are normalized to [0, 1], by default True
        Set to False if the images are in the range [0, 255].
    """

    def __init__(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        normalize: bool = True,
        subset_size: int = 250,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_runs: int = 1,  # unused but for consistency
    ):
        super(KIDMetricEvaluator, self).__init__(
            dataloader_train, dataloader_test, device=device
        )
        n_data_train = len(dataloader_train.dataset)
        n_data_test = len(dataloader_test.dataset)
        if n_data_train < subset_size or n_data_test < subset_size:
            LOGGER.warning(
                f"Warning: Subset size {subset_size} is larger than available data. Reducing subset size to minimum of train and test datasets."
            )
        subset_size = min(subset_size, n_data_train, n_data_test)
        self.kid_metric = (
            KernelInceptionDistance(normalize=normalize, subset_size=subset_size)
            .to(self.device)
            .to(torch.float)
        )

    def forward(self):
        self.kid_metric.reset()
        pbar = tqdm(
            total=len(self.dataloader_train) + len(self.dataloader_test),
            desc="Computing KID",
            unit="batches",
        )
        for i_train in self.dataloader_train:
            x_train, *_ = i_train
            x_train = x_train.to(self.device)
            # expects RGB images
            if x_train.shape[1] == 1:
                x_train = x_train.repeat(1, 3, 1, 1)
            self.kid_metric.update(x_train, real=True)
            pbar.update(1)
        for i_test in self.dataloader_test:
            x_test, *_ = i_test
            x_test = x_test.to(self.device)
            if x_test.shape[1] == 1:
                x_test = x_test.repeat(1, 3, 1, 1)
            self.kid_metric.update(x_test, real=False)
            pbar.update(1)
        pbar.close()
        score, std = self.kid_metric.compute()
        return score.item()


class FIDMetricEvaluator(MetricEvaluator):
    """
    Compute Frechet Inception Distance (FID) between images.

    Parameters
    ----------
    dataloader_train : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    dataloader_test : torch.utils.data.DataLoader
        DataLoader for the testing dataset.
    normalize : bool, optional
        Set to True if the images are normalized to [0, 1], by default True
        Set to False if the images are in the range [0, 255].
    """

    def __init__(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        normalize: bool = True,
        n_runs: int = 1,  # unused but for consistency
    ):
        super(FIDMetricEvaluator, self).__init__(
            dataloader_train, dataloader_test, device=device, n_runs=n_runs
        )
        self.device = device
        self.fid_metric = (
            FrechetInceptionDistance(normalize=normalize).to(self.device).to(torch.float)
        )

    def forward(self):
        self.fid_metric.reset()
        pbar = tqdm(
            total=len(self.dataloader_train) + len(self.dataloader_test),
            desc="Computing FID",
            unit="batches",
        )
        for i_train in self.dataloader_train:
            x_train, *_ = i_train
            x_train = x_train.to(self.device)
            # expects RGB images
            if x_train.shape[1] == 1:
                x_train = x_train.repeat(1, 3, 1, 1)
            self.fid_metric.update(x_train, real=True)
            pbar.update(1)
        for i_test in self.dataloader_test:
            x_test, *_ = i_test
            x_test = x_test.to(self.device)
            if x_test.shape[1] == 1:
                x_test = x_test.repeat(1, 3, 1, 1)
            self.fid_metric.update(x_test, real=False)
            pbar.update(1)
        pbar.close()

        return self.fid_metric.compute().item()


@torch.no_grad()
def forward_encoder(img_loader: DataLoader, encoder: torch.nn.Module, device: torch.device):
    """
    Forward pass through the encoder to extract features from images.
    Args:
        img_loader (DataLoader): DataLoader for images (3, H, W).
        encoder (torch.nn.Module): Pre-trained encoder model.
        device (torch.device): Device to perform calculations on.
    Returns:
        torch.Tensor: Encoded features. (N, D)
    """
    dtype = next(encoder.parameters()).dtype
    encoder.eval()
    all_features = []
    for img, *_ in tqdm(img_loader, desc="Extracting features", unit="batches"):
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        img = img.to(device).to(dtype)
        features = encoder(img)
        all_features.append(features.cpu())
    all_features = torch.cat(all_features, dim=0)
    return all_features


class PrecisionRecallEvaluator(MetricEvaluator):
    """
    Compute Precision and Recall between images using manifold estimation.
    "Improved Precision and Recall Metric for Assessing Generative Models"

    Formally for images I_train, I_test we compute
    Precision = fraction of I_test within the manifold of I_train
    Recall = fraction of I_train within the manifold of I_test

    Parameters
    ----------
    dataloader_train : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    dataloader_test : torch.utils.data.DataLoader
        DataLoader for the testing dataset.
    encoder : torch.nn.Module, optional
        Pre-trained encoder model to extract features, by default None
    k : int, optional
        Number of nearest neighbors for manifold estimation, by default 3
    device : torch.device, optional
        Device to perform calculations on, by default torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """

    def __init__(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        encoder: torch.nn.Module = None,
        k: int = 3,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(PrecisionRecallEvaluator, self).__init__(dataloader_train, dataloader_test)
        self.encoder = encoder
        self.k = k
        self.device = device

    def manifold_estimate(
        self, phi_a: torch.Tensor, phi_b: torch.Tensor, chunk_size: int = 200
    ):
        """
        Estimate the manifold form phi_a and computes the number
        of points from phi_b inside it.

        Args:
            phi_a : (N, D) : points
            phi_b : (M, D) : points
        Returns:
            score : float : score
        """
        # dist = torch.cdist(phi_a, phi_a, p=2)  # (N, N)
        # r_phi, _ = torch.topk(dist, k=self.k + 1, dim=1, largest=False)  # (N, k+1)
        # r_phi = r_phi[:, -1]  # (N, )

        # dst_to_true = torch.cdist(phi_a, phi_b, p=2)  # (N, M)
        # inside = (dst_to_true <= r_phi[:, None]).float()  # (N, M)
        # valid_pts = torch.any(inside, dim=0)  # (M, )
        # score = torch.sum(valid_pts) / valid_pts.shape[0]
        # return score.cpu().item()

        N = phi_a.shape[0]
        chunk_size = min(chunk_size, N)
        r_phi = torch.zeros(N)

        for start_idx in range(0, N, chunk_size):
            chunk = phi_a[start_idx : start_idx + chunk_size]
            dst_chunk = torch.cdist(chunk, phi_a, p=2)  # (chunk_size, N)
            r_chunk, _ = torch.topk(dst_chunk, k=self.k + 1, dim=1, largest=False)
            r_phi[start_idx : start_idx + chunk_size] = r_chunk[:, -1]

        M = phi_b.shape[0]
        chunk_size = min(chunk_size, M)
        valid_count = 0
        for start_idx in range(0, M, chunk_size):
            chunk = phi_b[start_idx : start_idx + chunk_size]
            dst_chunk = torch.cdist(phi_a, chunk, p=2)  # (N, chunk_size)
            inside = (dst_chunk <= r_phi[:, None]).float()
            valid_pts = torch.any(inside, dim=0)  # (chunk_size,)
            valid_count += torch.sum(valid_pts).cpu().item()

        score = valid_count / M
        return score

    def forward(self):
        if self.encoder is None:
            self.encoder = FIDInceptionV3()
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        phi_real = forward_encoder(self.dataloader_train, self.encoder, device=self.device)
        phi_fake = forward_encoder(self.dataloader_test, self.encoder, device=self.device)
        precision = self.manifold_estimate(phi_real, phi_fake)
        recall = self.manifold_estimate(phi_fake, phi_real)
        return precision, recall


class FWDMetricEvaluator(MetricEvaluator):
    """
    Compute FWD metric between images.
    See 'Fréchet Wavelet Distance: A Domain-Agnostic Metric for Image Generation'
    """

    def __init__(
        self,
        dataloader_train,
        dataloader_test,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_runs=2,
    ):
        super().__init__(dataloader_train, dataloader_test, device, n_runs)

    @torch.no_grad
    def forward(self):
        mu_0, sigma_0 = compute_packet_statistics(
            self.dataloader_train, wavelet="haar", max_level=4, log_scale=False
        )
        mu_1, sigma_1 = compute_packet_statistics(
            self.dataloader_test, wavelet="haar", max_level=4, log_scale=False
        )
        fwd = _compute_avg_frechet_distance(mu_0, mu_1, sigma_0, sigma_1)
        return fwd


class AEDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_geodesic: GeodesicEvalRiemannConfig, train: bool = True):
        self.cfg_geodesic = cfg_geodesic
        self.device = cfg_geodesic.device
        self.encoder = cfg_geodesic.encoder.to(self.device)
        self.decoder = cfg_geodesic.decoder.to(self.device)
        self.train = train
        if self.train:
            self.dataloader = cfg_geodesic.dataloader_train
        else:
            self.dataloader = cfg_geodesic.dataloader_test

    def __len__(self):
        return len(self.dataloader.dataset)

    def __getitem__(self, idx):
        data = self.dataloader.dataset[idx]
        x, t = data
        x = x.to(self.device).unsqueeze(0)
        z, _ = self.encoder(x)

        x_hat = self.decoder(z).squeeze(0)
        return x_hat, 0


class InterpolantsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        decoder: torch.nn.Module,
        export_path: Path,
        train: bool = True,
        transform=None,
        latent_name: str = "latent_geodesics",
        t_range: tuple[int, int] = None,
    ):
        self.decoder = decoder
        self.decoder.eval()
        self.device = next(decoder.parameters()).device
        self.export_path = export_path
        self.train = train
        self.transform = transform
        self.t_range = t_range

        latent_name = f"{latent_name}_{'train' if self.train else 'test'}.pth"
        self.latent_path = self.export_path / latent_name
        self.latent_traj = torch.load(self.latent_path, weights_only=True)  # (Num_traj, T, D)
        if t_range is not None:
            t_start, t_end = t_range
            self.latent_traj = self.latent_traj[:, t_start : t_end + 1, :]
        self.latent_traj = rearrange(self.latent_traj, "n t d -> (n t) d")  # (Num_traj * T, D)

        self.check_data_integrity()

    def check_data_integrity(self):
        # assert not self.latent_traj.isnan().any(), "Latent trajectory contains NaN values"
        has_nan = self.latent_traj.isnan().any()
        if has_nan:
            num_nan = self.latent_traj.isnan().sum().item()
            total_elements = self.latent_traj.numel()
            nan_percentage = (num_nan / total_elements) * 100
            LOGGER.warning(
                f"Latent trajectory contains {num_nan} NaN values out of {total_elements} elements ({nan_percentage:.2f}%)."
            )

    def __len__(self) -> int:
        return self.latent_traj.shape[0]

    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        z = self.latent_traj[idx].unsqueeze(0).to(self.device)
        x_pred = self.decoder(z).squeeze(0)
        if self.transform:
            x_pred = self.transform(x_pred)
        return x_pred, 0  # return a tuple for compatibility


ALL_METRICS: dict[str, MetricEvaluator] = {
    "fid": FIDMetricEvaluator,
    "kid": KIDMetricEvaluator,
    # "clip": CLIPMetricEvaluator,
    # "precision_recall": PrecisionRecallEvaluator,
    "fwd": FWDMetricEvaluator,
}


def eval_ae_model(cfg: GeodesicEvalRiemannConfig):
    LOGGER.info("Evaluating Auto-Encoder reconstruction quality...")
    ae_dataset_train = AEDataset(cfg, train=True)
    ae_dataset_test = AEDataset(cfg, train=False)

    ae_dataloader_train = torch.utils.data.DataLoader(
        ae_dataset_train,
        batch_size=cfg.solver_params.batch_size,
        shuffle=True,
    )
    ae_dataloader_test = torch.utils.data.DataLoader(
        ae_dataset_test,
        batch_size=cfg.solver_params.batch_size,
        shuffle=True,
    )
    res = {}
    for metric_name, metric_class in ALL_METRICS.items():
        LOGGER.info(f"  Evaluating {metric_name}...")
        try:
            metric_evaluator_train = metric_class(
                dataloader_train=cfg.dataloader_train,
                dataloader_test=ae_dataloader_train,
                device=cfg.device,
            )
            score_train = metric_evaluator_train()

            metric_evaluator_test = metric_class(
                dataloader_train=cfg.dataloader_train,
                dataloader_test=ae_dataloader_test,
                device=cfg.device,
            )
            score_test = metric_evaluator_test()

            res[f"{metric_name}_ae_train"] = score_train
            res[f"{metric_name}_ae_test"] = score_test
        except Exception as e:
            LOGGER.error(f"Error evaluating {metric_name}: {e}")
    return res


def export_metrics_results(
    cfg: GeodesicEvalRiemannConfig, scores: dict[str, float], t_range: tuple[int, int] = None
):
    filename = f"metrics_results"
    if t_range is not None:
        t_start, t_end = t_range
        filename += f"_t{t_start}_{t_end}"
    metrics_path = cfg.export_path / f"{filename}.json"
    with open(metrics_path, "w") as f:
        json.dump(scores, f, indent=4)


def evaluate_dataloaders(
    dataloader_reference: DataLoader,
    dataloader_generated: DataLoader,
    device: torch.device,
    train: bool = True,
):
    LOGGER.info(
        f"Evaluating {'train' if train else 'test'} geodesics quality. n = {len(dataloader_generated.dataset)}"
    )
    scores = {}
    for metric_name, metric_class in ALL_METRICS.items():
        LOGGER.info(f"  Evaluating {'train' if train else 'test'} {metric_name}...")
        try:
            metric_evaluator_train = metric_class(
                dataloader_train=dataloader_reference,
                dataloader_test=dataloader_generated,
                device=device,
            )
            score_train = metric_evaluator_train()
            key = f"{metric_name}_{'train' if train else 'test'}"
            scores[key] = score_train
            LOGGER.info(f"    {key}: {score_train}")
        except Exception as e:
            LOGGER.error(f"Error evaluating {metric_name}: {e}")
    return scores


def evaluate_framework(cfg: GeodesicEvalRiemannConfig):
    """
    Evaluate a framework by first retrieving the latent geodesics. Then decoding them.
    And finally evaluating the quality of the decoded geodesics using the dataloaders and metric evaluators.
    """
    start_time = datetime.now()
    inter_dataset_test = InterpolantsDataset(cfg.decoder, cfg.export_path, train=False)
    inter_dataloader_test = torch.utils.data.DataLoader(
        inter_dataset_test,
        batch_size=cfg.solver_params.batch_size,
        shuffle=True,
    )
    scores_test = evaluate_dataloaders(
        dataloader_reference=cfg.reference_dataloader_train,
        dataloader_generated=inter_dataloader_test,
        device=cfg.device,
        train=False,
    )
    export_metrics_results(cfg, scores_test)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    LOGGER.info(
        f"Image quality evaluation completed in {elapsed_time} for config {cfg.export_path}"
    )
    return scores_test


def check_extrapolation_exists(cfg: ExtrapolationEvalRiemannConfig) -> bool:
    """
    Check if the extrapolated geodesics already exist in the export path.
    """
    test_path = cfg.export_path / "extrapolated_traj_test.pth"
    return test_path.exists()


def evaluate_extrapolated_framework(cfg: ExtrapolationEvalRiemannConfig):
    """
    Evaluate a framework by first retrieving the latent chains. Then decoding them.
    And finally evaluating the quality of the decoded chains using the dataloaders and metric evaluators.
    """
    if not check_extrapolation_exists(cfg):
        LOGGER.error(
            "Extrapolated trajectories not found. Please run the extrapolation evaluation script first to generate them."
        )
        raise FileNotFoundError(
            "Extrapolated trajectories not found. Please run the extrapolation evaluation script first to generate them."
        )
    start_time = datetime.now()

    T = cfg.T
    # t_ranges = [(0, T // 4), (T // 4, T // 2), (T // 2, 3 * T // 4), (3 * T // 4, T)]
    t_ranges = [(0, T)]
    for t_range in t_ranges:
        LOGGER.info(f"Evaluating extrapolated trajectories for t range {t_range}...")
        inter_dataset_test = InterpolantsDataset(
            cfg.decoder,
            cfg.export_path,
            train=False,
            t_range=t_range,
            latent_name="extrapolated_traj",
        )
        inter_dataloader_test = torch.utils.data.DataLoader(
            inter_dataset_test,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        scores_test = evaluate_dataloaders(
            dataloader_reference=cfg.reference_dataloader_train,
            dataloader_generated=inter_dataloader_test,
            device=cfg.device,
            train=False,
        )
        export_metrics_results(cfg, scores_test, t_range=t_range)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    LOGGER.info(
        f"Image quality evaluation completed in {elapsed_time} for config {cfg.export_path}"
    )
    return scores_test
