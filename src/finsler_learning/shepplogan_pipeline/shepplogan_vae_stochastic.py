from pathlib import Path

import torch

from ..base_vae_trainer import VAEConfig, vae_trainer
from ..models import StochasticVAE
from .shepplogan_utils import (
    DEVICE,
    IMG_SIZE,
    LOGGER,
    DatasetConfig,
    get_shepplogan_dataloaders,
    get_stochastic_models,
)
from .shepplogan_vae import CondVAELoss


################################################################
# VAE training
################################################################
def train_vae_log_var_decoder_branch(
    latent_dim: int,
    export_path: Path,
    lr: float,
    weight_decay: float,
    beta: float,
    lbda: float,
    lpip_weight: float,
    n_epochs: int,
    batch_size: int,
    label_noise: float,
    K: int,
    kappa: float,
    device: torch.device = DEVICE,
) -> tuple[torch.nn.Module, list, list]:

    LOGGER.info("Training the log_var_branch of the decoder of a StochasticVAE model...")
    LOGGER.info(f"Exporting model to {export_path / 'stochastic_vae'}")

    dl_config = DatasetConfig(batch_size=batch_size, label_noise=label_noise)
    shepplogan_train_loader, shepp_logan_test_loader = get_shepplogan_dataloaders(
        dataset_config=dl_config, export_path=export_path
    )

    if K is None:
        K = len(shepplogan_train_loader.dataset)

    model: StochasticVAE = get_stochastic_models(
        latent_dim=latent_dim,
        save_folder=export_path,
        dataloader=shepplogan_train_loader,
        K=K,
        kappa=kappa,
    ).to(device)

    # Train only the log_var_branch of the decoder.
    # Not the encoder and the mu_branch of the decoder
    optim = torch.optim.Adam(
        model.decoder.log_var_branch.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim, T_max=n_epochs, eta_min=1e-6
    )
    loss_fn = CondVAELoss(beta=beta, lbda=lbda, lpip_weight=lpip_weight)
    task_config = VAEConfig(
        n_epochs=n_epochs,
        dataloader=shepplogan_train_loader,
        dataloader_test=shepp_logan_test_loader,
        model=model,
        optimizer=optim,
        criterion=loss_fn,
        export_path=export_path / "stochastic_vae",
        img_shape=(1, IMG_SIZE, IMG_SIZE),
        scheduler=scheduler,
        use_beta_scheduler=True,
        device=device,
    )
    model, train_loss_list, test_loss_list = vae_trainer(task_config)
    return model, train_loss_list, test_loss_list
