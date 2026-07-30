import torch
import logging

from torch.utils.data import DataLoader

from pathlib import Path
from dataclasses import dataclass
from json import dump
from tqdm import tqdm

from geodesic_toolbox import (
    CentroidsCometric,
    CoMetric,
    PullBackCometric,
    IdentityCoMetric,
    LANDCometric,
)

from finsler_learning import BaseVAE, VAE, SheppLoganDataset, WarpedDecoder

#################################################################
# Define constants
#################################################################
IMG_SIZE = 256
N_SAMPLES_TRAIN = 1000
N_SAMPLES_TEST = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger()


VALID_COMETRIC_TYPES = [
    # "vae", "pullback", "euclidean",
    "warped_pullback",
    # "land"
]


@dataclass
class SheppLoganPipelineConfig:
    # Image regression parameters
    n_epochs_regressor: int
    b_size_regressor: int
    lr_regressor: float
    wd_regressor: float

    # VAE training parameters
    n_epochs_vae: int
    b_size_vae: int
    lr_vae: float
    wd_vae: float
    beta: float
    lbda: float
    lpip_weight: float

    # Randers training parameters
    n_neighbors: int
    margin: float
    n_epochs_randers: int
    b_size_randers: int
    use_dummy_accuracy: bool
    lr_randers: float
    wd_randers: float
    omega_strength: float

    # Evaluation of monotonicity parameters
    b_size_evaluation: int
    eta: float
    alpha: float
    ema_moment: float
    n_pts: int  # number of points to sample along the geodesic for evaluation

    # Training flags
    skip_regressor: bool
    skip_vae: bool
    skip_randers: bool
    eval_monotonicity: bool

    # Global parameters
    export_path: Path
    latent_dim: int
    label_noise: float = 0.0
    K_cometric: int = -1
    IMG_SIZE: int = IMG_SIZE
    device: torch.device = DEVICE
    eval_quality: bool = True
    eval_extrapolation: bool = False
    eval_regressor: bool = False
    no_symmetric: bool = False


@dataclass
class DatasetConfig:
    batch_size: int = 32
    n_workers: int = 0
    smoothing: bool = True
    normalize_area: bool = False
    img_size: int = IMG_SIZE
    num_samples_train: int = N_SAMPLES_TRAIN
    num_samples_val: int = N_SAMPLES_TEST
    key: str = "ventricle1"
    label_noise: float = 0.0


def export_shepplogan_config(config: SheppLoganPipelineConfig):
    cfg_dict = {}
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        if isinstance(value, Path):
            cfg_dict[field] = str(value)
        elif isinstance(value, torch.device):
            cfg_dict[field] = str(value)
        else:
            cfg_dict[field] = value

    config_file = config.export_path / "shepplogan_pipeline_config.json"
    with open(config_file, "w") as f:
        dump(cfg_dict, f, indent=4)
    LOGGER.info(f"SheppLogan Pipeline Configuration exported to {config_file}")


###############################################################
# Utils functions
###############################################################
def get_models(latent_dim: int) -> VAE:
    model = VAE(latent_dim=latent_dim, affine=True)
    return model


def try_load_pretrained_models(save_folder: Path, model: BaseVAE):
    if (save_folder / "encoder.pth").exists():
        state_dict_enc = torch.load(save_folder / "encoder.pth", weights_only=True)
        model.encoder.load_state_dict(state_dict_enc)
        LOGGER.info(f"Loaded encoder from {save_folder / 'encoder.pth'}")
    if (save_folder / "decoder.pth").exists():
        state_dict_dec = torch.load(save_folder / "decoder.pth", weights_only=True)
        model.decoder.load_state_dict(state_dict_dec)
        LOGGER.info(f"Loaded decoder from {save_folder / 'decoder.pth'}")


def get_shepplogan_dataloaders(
    dataset_config: DatasetConfig,
    export_path: Path = None,
):
    train_dataset = SheppLoganDataset(
        img_size=dataset_config.img_size,
        num_samples=dataset_config.num_samples_train,
        smoothing=dataset_config.smoothing,
        export_path=export_path,
        train=True,
        normalize_area=dataset_config.normalize_area,
        label_noise=dataset_config.label_noise,
    )
    test_dataset = SheppLoganDataset(
        img_size=dataset_config.img_size,
        num_samples=dataset_config.num_samples_val,
        export_path=export_path,
        train=False,
        normalize_area=dataset_config.normalize_area,
        label_noise=dataset_config.label_noise,
    )
    shepplogan_train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.n_workers,
        pin_memory=True,
    )
    shepplogan_test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.n_workers,
        pin_memory=True,
    )
    return shepplogan_train_loader, shepplogan_test_loader


@torch.no_grad()
def embed_dataset(
    encoder, dataloader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder.to(device)
    z_list, log_var_list = [], []
    labels_list = []
    for data in dataloader:
        x, labels = data
        x = x.to(device)
        mu, logvar = encoder(x)
        z_list.append(mu.cpu())
        labels_list.append(labels)
        log_var_list.append(logvar.cpu())
    z_all = torch.cat(z_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    log_var_all = torch.cat(log_var_list, dim=0)
    return z_all, labels_all, log_var_all


def get_cometric_vae(
    batch_size: int,
    latent_dim: int,
    save_folder: Path,
    K: int,
    device: torch.device = DEVICE,
) -> CentroidsCometric:
    # Don't bother with all the params as it's going to load generated data anyway
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    shepplogan_train_loader, _ = get_shepplogan_dataloaders(
        dataset_config=dl_config,
        export_path=save_folder,
    )

    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    z_all, _, log_var_all = embed_dataset(model.encoder, shepplogan_train_loader, device)

    log_var_all = log_var_all.clamp(min=-30, max=30)
    cometric_centroids = torch.exp(-2 * log_var_all)
    cometric = CentroidsCometric(
        centroids=z_all,
        cometric_centroids=cometric_centroids,
        K=K,
    )
    return cometric


def get_cometric_pullback(
    batch_size: int,
    latent_dim: int,
    save_folder: Path,
    K: int,
    b_size_pullback: int = 16,
    device: torch.device = DEVICE,
) -> CentroidsCometric:
    # Don't bother with all the params as it's going to load generated data anyway
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    shepplogan_train_loader, _ = get_shepplogan_dataloaders(
        dataset_config=dl_config, export_path=save_folder
    )
    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    centroids, _, _ = embed_dataset(model.encoder, shepplogan_train_loader, device)
    pullback_cometric = PullBackCometric(
        diffeo=model.decoder,
        reg_coef=0.0,  # We want exact pullback cometric, no regularization
        method="loop_jvp",  # We want exact pullback cometric, no finite difference
        chunk_size=1,
    ).to(device)
    centroids_cometric = []
    pbar = tqdm(
        range(0, len(centroids), b_size_pullback),
        desc="Computing pullback cometric",
        leave=False,
        total=len(centroids),
    )
    for b_idx in pbar:
        z = centroids[b_idx : b_idx + b_size_pullback].to(device)
        G_inv = pullback_cometric.cometric_tensor(z)
        centroids_cometric.append(G_inv)
        pbar.update(min(b_size_pullback, len(centroids) - b_idx) - 1)
    pbar.close()

    centroids_cometric = torch.cat(centroids_cometric, dim=0)
    cometric = CentroidsCometric(
        centroids=centroids, cometric_centroids=centroids_cometric, K=K
    )
    return cometric


def get_cometric_land(
    batch_size: int,
    latent_dim: int,
    save_folder: Path,
    K: int,
    device: torch.device = DEVICE,
):
    # Don't bother with all the params as it's going to load generated data anyway
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    shepplogan_train_loader, _ = get_shepplogan_dataloaders(
        dataset_config=dl_config,
        export_path=save_folder,
    )

    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    z_all, _, _ = embed_dataset(model.encoder, shepplogan_train_loader, device)
    cometric = LANDCometric(centroids=z_all, alpha=1, sigma=None, K=K)
    return cometric


def get_cometric_warped_pullback(
    batch_size: int,
    latent_dim: int,
    save_folder: Path,
    K: int,
    device: torch.device = DEVICE,
    beta: float = 1.0,
) -> PullBackCometric:
    # Don't bother with all the params as it's going to load generated data anyway
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    shepplogan_train_loader, _ = get_shepplogan_dataloaders(
        dataset_config=dl_config, export_path=save_folder
    )
    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    centroids, _, _ = embed_dataset(model.encoder, shepplogan_train_loader, device)

    warped_decoder = WarpedDecoder(
        decoder=model.decoder,
        centroids=centroids,
        beta=beta,
        K=K,
    ).to(device)

    pullback_cometric = PullBackCometric(
        diffeo=warped_decoder,
        reg_coef=1e-3,
        method="finite_difference",
        chunk_size=1,
    ).to(device)
    return pullback_cometric


@torch.no_grad()
def get_cometric(
    latent_dim: int,
    export_path: Path,
    k: int,
    batch_size: int,
    device: torch.device,
    cometric_type: str = "vae",
    b_size_pullback: int = 16,
) -> CoMetric:

    assert (
        cometric_type in VALID_COMETRIC_TYPES
    ), f"Invalid cometric_type: {cometric_type}. Must be one of {VALID_COMETRIC_TYPES}."

    if cometric_type == "vae":
        cometric = get_cometric_vae(
            batch_size=batch_size,
            latent_dim=latent_dim,
            save_folder=export_path,
            K=k,
            device=device,
        )
    elif cometric_type == "pullback":
        cometric = get_cometric_pullback(
            batch_size=batch_size,
            latent_dim=latent_dim,
            save_folder=export_path,
            K=k,
            b_size_pullback=b_size_pullback,
        )
    elif cometric_type == "euclidean":
        cometric = IdentityCoMetric()
    elif cometric_type == "land":
        cometric = get_cometric_land(
            batch_size=batch_size,
            latent_dim=latent_dim,
            save_folder=export_path,
            K=k,
            device=device,
        )
    elif cometric_type == "warped_pullback":
        cometric = get_cometric_warped_pullback(
            batch_size=batch_size,
            latent_dim=latent_dim,
            save_folder=export_path,
            K=k,
            device=device,
        )
    else:
        raise ValueError(
            f"Invalid cometric_type: {cometric_type}. Must be one of {VALID_COMETRIC_TYPES}."
        )
    return cometric.to(device)
