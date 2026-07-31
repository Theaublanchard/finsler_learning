import datetime

from finsler_learning.models import (
    HyperPlanPredictor,
    DummyAccuracyPredictor,
    AccuracyPredictor,
    Omega,
)
from finsler_learning.base_randers_learning import RandersLearningConfig, train_hyperplane
from finsler_learning.losses import MarginLoss

from .shepplogan_utils import *


#################################################################
# Randers training
#################################################################
def get_omega(latent_dim: int, cometric: CoMetric, use_dummy_accuracy: bool = False):
    phi = HyperPlanPredictor(
        latent_dim=latent_dim,
        hidden_dims=[latent_dim // 2, latent_dim // 2],
    )
    if use_dummy_accuracy:
        accuracy_predictor = DummyAccuracyPredictor()
    else:
        accuracy_predictor = AccuracyPredictor(
            latent_dim=latent_dim,
            hidden_dims=[latent_dim // 2, latent_dim // 4],
        )
    omega = Omega(direction=phi, amplitude=accuracy_predictor, cometric=cometric)
    return omega


def train_randers(
    export_path: Path,
    latent_dim: int,
    lr: float,
    wd: float,
    n_neighbors: int,
    margin: float,
    n_epochs: int,
    batch_size: int,
    use_dummy_accuracy: bool,
    cometric_type: str,
    k: int,
    device: torch.device = DEVICE,
):
    (export_path / f"randers_{cometric_type}").mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Training Randers metric with cometric type: {cometric_type}")

    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    shepplogan_train_loader, shepp_logan_test_loader = get_shepplogan_dataloaders(
        dataset_config=dl_config,
        export_path=export_path,
    )

    if k is None:
        k = len(shepplogan_train_loader.dataset)

    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(export_path, model)
    dtype = next(model.parameters()).dtype
    encoder = model.encoder.to(device)
    encoder.eval()
    omega = get_omega(latent_dim, IdentityCoMetric(), use_dummy_accuracy).to(device).to(dtype)

    cometric = get_cometric(
        latent_dim=latent_dim,
        export_path=export_path,
        k=k,
        batch_size=batch_size,
        device=device,
        cometric_type=cometric_type,
    )

    start_time = datetime.datetime.now()
    config = RandersLearningConfig(
        dataloader_train=shepplogan_train_loader,
        dataloader_test=shepp_logan_test_loader,
        encoder=encoder,
        omega=omega,
        loss_fn=MarginLoss(k=n_neighbors, margin=margin).to(device),
        n_epochs=n_epochs,
        export_path=export_path / f"randers_{cometric_type}",
        lr=lr,
        weight_decay=wd,
        device=device,
        cometric=cometric,
    )
    train_hyperplane(config)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    LOGGER.info(
        f"Randers training completed in: {elapsed_time} for cometric type: {cometric_type}"
    )
