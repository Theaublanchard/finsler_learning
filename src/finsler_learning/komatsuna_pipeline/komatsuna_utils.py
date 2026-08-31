import torch
import logging

from torch.utils.data import DataLoader
from torchvision.transforms import v2

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
    LANDRBFCometric,
    PBIG_Cometric_Gaussian,
)

from finsler_learning import (
    BaseStochasticVAE,
    VAE,
    StochasticVAE,
    KOMATSUNADataset,
    WarpedDecoder,
)

#################################################################
# Define constants
#################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger()

VALID_COMETRIC_TYPES = {
    "euclidean": {"needs_spline": False, "requires_stochastic_vae": False},
    "land_rbf": {"needs_spline": False, "requires_stochastic_vae": False},
    "land": {"needs_spline": False, "requires_stochastic_vae": False},
    "cov_centroids": {"needs_spline": False, "requires_stochastic_vae": False},
    "pullback_centroids": {"needs_spline": True, "requires_stochastic_vae": False},
    # "warped_pullback": {"needs_spline": True, "requires_stochastic_vae": False},
    # "rbf_pullback": {"needs_spline": True, "requires_stochastic_vae": True},
    # "pbig": {"needs_spline": True, "requires_stochastic_vae": True},
}


@dataclass
class KomatsunaPipelineConfig:
    # VAE training parameters
    n_epochs_vae: int
    b_size_vae: int
    lr_vae: float
    wd_vae: float
    beta: float
    lbda: float
    lpip_weight: float
    l1_weight: float

    # NNUNet training parameters
    n_epochs_nnunet: int

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
    skip_nnunet: bool
    skip_vae: bool
    skip_randers: bool
    eval_interpolation: bool
    eval_extrapolation: bool

    # Global parameters
    export_path: Path
    latent_dim: int
    K_cometric: int = None
    device: torch.device = DEVICE
    eval_quality_interpolation: bool = True
    eval_quality_extrapolation: bool = False
    eval_regressor: bool = False
    no_symmetric: bool = False


@dataclass
class DatasetConfig:
    batch_size: int = 32
    n_workers: int = 0
    return_annots: bool = False


def export_komatsuna_config(config: KomatsunaPipelineConfig):
    cfg_dict = {}
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        if isinstance(value, Path):
            cfg_dict[field] = str(value)
        elif isinstance(value, torch.device):
            cfg_dict[field] = str(value)
        else:
            cfg_dict[field] = value

    config_file = config.export_path / "komatsuna_pipeline_config.json"
    with open(config_file, "w") as f:
        dump(cfg_dict, f, indent=4)
    LOGGER.info(f"Komatsuna Pipeline Configuration exported to {config_file}")


###############################################################
# Utils functions
###############################################################
def get_models(latent_dim: int) -> VAE:
    model = VAE(latent_dim=latent_dim, affine=True, in_channels=3)
    return model


def get_stochastic_models(
    latent_dim: int,
    save_folder: Path,
    dataloader: torch.utils.data.DataLoader,
    K: int,
    kappa: float,
) -> StochasticVAE:
    base_model = VAE(latent_dim=latent_dim, affine=True, in_channels=3)
    try_load_pretrained_models(save_folder, base_model)
    data, _, _ = embed_dataset(base_model.encoder, dataloader, DEVICE)
    model = StochasticVAE(base_model=base_model, data=data, K=K, kappa=kappa)
    return model


def try_load_pretrained_models(save_folder: Path, model: BaseStochasticVAE):
    if (save_folder / "encoder.pth").exists():
        state_dict_enc = torch.load(save_folder / "encoder.pth", weights_only=True)
        model.encoder.load_state_dict(state_dict_enc)
        LOGGER.info(f"Loaded encoder from {save_folder / 'encoder.pth'}")
    if (save_folder / "decoder.pth").exists():
        state_dict_dec = torch.load(save_folder / "decoder.pth", weights_only=True)
        model.decoder.load_state_dict(state_dict_dec)
        LOGGER.info(f"Loaded decoder from {save_folder / 'decoder.pth'}")


def get_stochastic_vae_pretrained(
    save_folder: Path,
    latent_dim: int,
    dataloader: torch.utils.data.DataLoader,
    K: int,
    kappa: float,
) -> StochasticVAE:
    model = get_stochastic_models(
        latent_dim=latent_dim,
        save_folder=save_folder,
        dataloader=dataloader,
        K=K,
        kappa=kappa,
    )
    try_load_pretrained_models(save_folder / "stochastic_vae", model)
    return model


def _build_komatsuna_train_transform():
    transforms = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                )
            ],
            p=0.8,
        ),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.2,
        ),
        v2.Lambda(lambda x: x.clamp(0, 1)),
    ]
    return v2.Compose(transforms)


def get_komatsuna_dataloaders(
    dataset_config: DatasetConfig,
    export_path: Path = None,
    use_augmentation: bool = False,
):
    train_augment = _build_komatsuna_train_transform()
    train_dataset = KOMATSUNADataset(
        train=True,
        transform=train_augment if use_augmentation else None,
        export_path=export_path,
        return_annots=dataset_config.return_annots,
    )
    test_dataset = KOMATSUNADataset(
        train=False,
        transform=None,
        export_path=export_path,
        return_annots=dataset_config.return_annots,
    )
    komatsuna_train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.n_workers,
        pin_memory=True,
    )
    komatsuna_test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        num_workers=dataset_config.n_workers,
        pin_memory=True,
    )
    return komatsuna_train_loader, komatsuna_test_loader


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


def get_cometric_cov_centroids(
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
    komatsuna_train_loader, _ = get_komatsuna_dataloaders(
        dataset_config=dl_config,
        export_path=save_folder,
    )

    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    model.eval()
    z_all, _, log_var_all = embed_dataset(model.encoder, komatsuna_train_loader, device)

    log_var_all = log_var_all.clamp(min=-30, max=30)
    cometric_centroids = torch.exp(-2 * log_var_all)
    cometric = CentroidsCometric(
        centroids=z_all,
        cometric_centroids=cometric_centroids,
        K=K,
    )
    return cometric


# We can use no_grad because we
# use finite differences to compute the cometric, so we don't need gradients.
@torch.no_grad()
def get_cometric_pullback_centroids(
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
    komatsuna_train_loader, _ = get_komatsuna_dataloaders(
        dataset_config=dl_config, export_path=save_folder
    )
    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    model.eval()
    centroids, _, _ = embed_dataset(model.encoder, komatsuna_train_loader, device)
    pullback_cometric = PullBackCometric(
        diffeo=model.decoder,
        reg_coef=0.0,  # We want exact pullback cometric, no regularization
        method="finite_difference",  # We want exact pullback cometric, no finite difference
        chunk_size=8,
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
    komatsuna_train_loader, _ = get_komatsuna_dataloaders(
        dataset_config=dl_config,
        export_path=save_folder,
    )

    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    model.eval()
    z_all, _, _ = embed_dataset(model.encoder, komatsuna_train_loader, device)
    cometric = LANDCometric(centroids=z_all, alpha=1, K=K)
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
    komatsuna_train_loader, _ = get_komatsuna_dataloaders(
        dataset_config=dl_config, export_path=save_folder
    )
    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    model.eval()
    centroids, _, _ = embed_dataset(model.encoder, komatsuna_train_loader, device)

    warped_decoder = WarpedDecoder(
        decoder=model.decoder,
        centroids=centroids,
        beta=beta,
        K=K,
    ).to(device)

    pullback_cometric = PullBackCometric(
        diffeo=warped_decoder,
        method="finite_difference",
        chunk_size=16,
    ).to(device)
    return pullback_cometric


def get_cometric_land_rbf(
    batch_size: int,
    latent_dim: int,
    save_folder: Path,
    K: int,
    device: torch.device = DEVICE,
) -> LANDRBFCometric:
    # Don't bother with all the params as it's going to load generated data anyway
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    komatsuna_train_loader, _ = get_komatsuna_dataloaders(
        dataset_config=dl_config, export_path=save_folder
    )
    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(save_folder, model)
    model.eval()
    centroids, _, _ = embed_dataset(model.encoder, komatsuna_train_loader, device)

    land_rbf_cometric = LANDRBFCometric(
        data=centroids,
        K=K,
        kappa=1.0,
        learn_weights=True,
    ).to(device)
    return land_rbf_cometric


def get_cometric_rbf_pullback(
    batch_size: int,
    latent_dim: int,
    save_folder: Path,
    K: int,
    device: torch.device = DEVICE,
) -> PullBackCometric:
    # Don't bother with all the params as it's going to load generated data anyway
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    komatsuna_train_loader, _ = get_komatsuna_dataloaders(
        dataset_config=dl_config, export_path=save_folder
    )
    model = get_stochastic_vae_pretrained(
        save_folder=save_folder,
        latent_dim=latent_dim,
        dataloader=komatsuna_train_loader,
        K=K,
        kappa=1.0,
    ).to(device)
    model.eval()

    pb_decoder_mu = PullBackCometric(
        diffeo=model.decoder.mu_branch,
        method="finite_difference",
        reg_coef=0.0,
    )
    pb_decoder_logvar = PullBackCometric(
        diffeo=model.decoder.std_branch,
        method="finite_difference",
        reg_coef=0.0,
    )

    cometric = pb_decoder_mu + pb_decoder_logvar
    return cometric.to(device)


def get_cometric_pbig(
    batch_size: int,
    latent_dim: int,
    save_folder: Path,
    K: int,
    device: torch.device = DEVICE,
) -> PBIG_Cometric_Gaussian:
    # Don't bother with all the params as it's going to load generated data anyway
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    komatsuna_train_loader, _ = get_komatsuna_dataloaders(
        dataset_config=dl_config, export_path=save_folder
    )
    model = get_stochastic_vae_pretrained(
        save_folder=save_folder,
        latent_dim=latent_dim,
        dataloader=komatsuna_train_loader,
        K=K,
        kappa=1.0,
    ).to(device)
    model.eval()

    cometric = PBIG_Cometric_Gaussian(
        decoder=model.decoder.complete_forward, epsilon=1e-4, rho=1e-4
    )
    return cometric.to(device)


# @torch.inference_mode()
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
        cometric_type in VALID_COMETRIC_TYPES.keys()
    ), f"Invalid cometric_type: {cometric_type}. Must be one of {VALID_COMETRIC_TYPES.keys()}."

    if cometric_type == "cov_centroids":
        cometric = get_cometric_cov_centroids(
            batch_size=batch_size,
            latent_dim=latent_dim,
            save_folder=export_path,
            K=k,
            device=device,
        )
    elif cometric_type == "pullback_centroids":
        cometric = get_cometric_pullback_centroids(
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
    elif cometric_type == "land_rbf":
        cometric = get_cometric_land_rbf(
            batch_size=batch_size,
            latent_dim=latent_dim,
            save_folder=export_path,
            K=k,
            device=device,
        )
    elif cometric_type == "rbf_pullback":
        cometric = get_cometric_rbf_pullback(
            batch_size=batch_size,
            latent_dim=latent_dim,
            save_folder=export_path,
            K=k,
            device=device,
        )
    elif cometric_type == "pbig":
        cometric = get_cometric_pbig(
            batch_size=batch_size,
            latent_dim=latent_dim,
            save_folder=export_path,
            K=k,
            device=device,
        )
    else:
        raise NotImplementedError(
            f"Cometric type {cometric_type} not implemented. Must be one of {VALID_COMETRIC_TYPES.keys()}."
        )
    return cometric.to(device)
