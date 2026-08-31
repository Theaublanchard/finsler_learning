import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from einops import rearrange

from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .base_vae_trainer import (
    export_results_vae,
    visualize_reconstruction,
    _set_criterion_beta,
    _get_scheduled_beta,
    _append_component_history,
    export_loss_components,
)
from .losses import NaNLossException, Losse
from .models import BaseSphericalVAE


@dataclass
class SphericalVAEConfig:
    n_epochs: int
    dataloader: DataLoader
    dataloader_test: DataLoader
    model: BaseSphericalVAE
    optimizer: optim.Optimizer
    criterion: callable
    export_path: str
    img_shape: tuple[int, ...]  # Shape of the input images for visualization
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_progress_bar: bool = False  # If True, shows a progress bar for each epoch
    scheduler: optim.lr_scheduler.LRScheduler | None = None
    export_interval: int = 5  # Interval (in epochs) to export results and visualizations
    use_beta_scheduler: bool = False
    beta_start: float = 0.0
    beta_end: float = 1.0


def train_loop_vae(
    epoch: int,
    dataloader: DataLoader,
    model: BaseSphericalVAE,
    optimizer: optim.Optimizer,
    criterion: Losse,
    device: torch.device,
    epoch_progress_bar: bool = False,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> float | tuple[float, dict[str, float]]:
    model.train()
    loss_list_train_epoch = []

    if epoch_progress_bar:
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Train Epoch {epoch + 1}",
            unit="batch",
        )
    else:
        pbar = enumerate(dataloader)

    component_sums: dict[str, float] = {}
    n_component_batches = 0

    for i, data in pbar:
        x, y = data
        x = x.to(device)
        y = y.to(device)

        x_hat, mu, rho = model(x)

        optimizer.zero_grad()
        loss = criterion(x_hat, x, mu, rho)

        if torch.isnan(loss).any():
            raise NaNLossException(f"NaN loss encountered at epoch {epoch}, batch {i}. ")

        loss.backward()
        optimizer.step()

        loss_list_train_epoch.append(loss)

        for key, value in criterion.curr_it_component.items():
            if torch.is_tensor(value):
                value_float = float(value.detach().item())
            else:
                value_float = float(value)
            component_sums[key] = component_sums.get(key, 0.0) + value_float
        n_component_batches += 1

    if scheduler is not None:
        scheduler.step()

    loss_list_train_epoch = [l.item() for l in loss_list_train_epoch]
    if epoch > 0:
        loss_train_epoch = np.mean(loss_list_train_epoch)
    else:
        loss_train_epoch = np.mean(
            loss_list_train_epoch[-5:]
        )  # Use last 5 batches for first epoch

    component_means = {
        key: value / max(n_component_batches, 1) for key, value in component_sums.items()
    }
    return loss_train_epoch, component_means


@torch.no_grad()
def test_loop_vae(
    epoch: int,
    dataloader_test: DataLoader,
    model: BaseSphericalVAE,
    criterion: Losse,
    device: torch.device,
    epoch_progress_bar: bool = False,
) -> float | tuple[float, dict[str, float]]:
    model.eval()
    loss_list_test_epoch = []

    if epoch_progress_bar:
        pbar = tqdm(
            enumerate(dataloader_test),
            total=len(dataloader_test),
            desc=f"Test Epoch {epoch + 1}",
            unit="batch",
        )
    else:
        pbar = enumerate(dataloader_test)
    component_sums: dict[str, float] = {}
    n_component_batches = 0

    for i, data in pbar:
        x, y = data
        x = x.to(device)
        y = y.to(device)

        x_hat, mu, rho = model(x)
        loss = criterion(x_hat, x, mu, rho)

        if torch.isnan(loss).any():
            raise NaNLossException(f"NaN loss encountered at epoch {epoch}, batch {i}. ")

        loss_list_test_epoch.append(loss)
        for key, value in criterion.curr_it_component.items():
            if torch.is_tensor(value):
                value_float = float(value.detach().item())
            else:
                value_float = float(value)
            component_sums[key] = component_sums.get(key, 0.0) + value_float
        n_component_batches += 1

    loss_list_test_epoch = [l.item() for l in loss_list_test_epoch]
    loss_test_epoch = np.mean(loss_list_test_epoch)

    component_means = {
        key: value / max(n_component_batches, 1) for key, value in component_sums.items()
    }
    return loss_test_epoch, component_means


def vae_trainer(vae_config: SphericalVAEConfig) -> nn.Module:
    export_path = Path(vae_config.export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    loss_list_train = []
    loss_list_test = []
    component_history_train: dict[str, list[float]] = {}
    component_history_test: dict[str, list[float]] = {}

    vae_config.model.to(vae_config.device)

    pbar = tqdm(range(vae_config.n_epochs), desc="Training VAE")
    for epoch in pbar:
        current_beta = None
        if vae_config.use_beta_scheduler:
            current_beta = _get_scheduled_beta(
                epoch=epoch,
                n_epochs=vae_config.n_epochs,
                beta_start=vae_config.beta_start,
                beta_end=vae_config.beta_end,
            )
            _set_criterion_beta(vae_config.criterion, current_beta)

        loss_epoch_train, train_components = train_loop_vae(
            epoch=epoch,
            dataloader=vae_config.dataloader,
            model=vae_config.model,
            optimizer=vae_config.optimizer,
            criterion=vae_config.criterion,
            device=vae_config.device,
            epoch_progress_bar=vae_config.epoch_progress_bar,
            scheduler=vae_config.scheduler,
        )
        loss_epoch_test, test_components = test_loop_vae(
            epoch=epoch,
            dataloader_test=vae_config.dataloader_test,
            model=vae_config.model,
            criterion=vae_config.criterion,
            device=vae_config.device,
            epoch_progress_bar=vae_config.epoch_progress_bar,
        )

        train_components["beta"] = vae_config.criterion.beta
        test_components["beta"] = vae_config.criterion.beta

        if epoch == 0:
            component_history_train = {key: [value] for key, value in train_components.items()}
            component_history_test = {key: [value] for key, value in test_components.items()}
        else:
            _append_component_history(component_history_train, train_components)
            _append_component_history(component_history_test, test_components)

        pbar.set_postfix(
            train_loss=loss_epoch_train,
            test_loss=loss_epoch_test,
        )

        loss_list_train.append(loss_epoch_train)
        loss_list_test.append(loss_epoch_test)

        if epoch % vae_config.export_interval == 0 and epoch > 0:
            export_results_vae(
                export_path,
                vae_config.model,
                loss_list_train,
                loss_list_test,
                component_history_train=component_history_train,
                component_history_test=component_history_test,
            )

        if epoch % vae_config.export_interval == 0:
            visualize_reconstruction(
                vae_config.model,
                vae_config.dataloader_test,
                vae_config.device,
                export_path,
                img_shape=vae_config.img_shape,
                filename="reconstruction_vae_test.png",
            )
            visualize_reconstruction(
                vae_config.model,
                vae_config.dataloader,
                vae_config.device,
                export_path,
                img_shape=vae_config.img_shape,
                filename="reconstruction_vae_train.png",
            )

    logger = logging.getLogger()
    logger.info(f"VAE training complete. Results exported to: {export_path}")
    if vae_config.use_beta_scheduler:
        logger.info(
            "Beta scheduler enabled: beta linearly varied from "
            + f"{vae_config.beta_start:.6f} to {vae_config.beta_end:.6f}"
        )
    logger.info(f"Final Train Loss: {loss_list_train[-1]:.6f}")
    logger.info(f"Final Test Loss: {loss_list_test[-1]:.6f}")
    export_results_vae(
        export_path,
        vae_config.model,
        loss_list_train,
        loss_list_test,
        component_history_train=component_history_train,
        component_history_test=component_history_test,
    )
    visualize_reconstruction(
        vae_config.model,
        vae_config.dataloader_test,
        vae_config.device,
        export_path,
        img_shape=vae_config.img_shape,
        filename="reconstruction_vae_test.png",
    )
    visualize_reconstruction(
        vae_config.model,
        vae_config.dataloader,
        vae_config.device,
        export_path,
        img_shape=vae_config.img_shape,
        filename="reconstruction_vae_train.png",
    )

    return vae_config.model, loss_list_train, loss_list_test
