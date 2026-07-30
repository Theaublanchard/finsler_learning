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

from .losses import NaNLossException, Losse
from .models import BaseVAE


@dataclass
class VAEConfig:
    n_epochs: int
    dataloader: DataLoader
    dataloader_test: DataLoader
    model: BaseVAE
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


def _set_criterion_beta(criterion: callable, beta: float) -> bool:
    """Set beta on criterion when supported.

    Supports losses exposing a direct `beta` attribute.
    """
    if hasattr(criterion, "beta"):
        criterion.beta = beta
        return True
    return False


def _get_scheduled_beta(
    epoch: int, n_epochs: int, beta_start: float, beta_end: float
) -> float:
    if n_epochs <= 1:
        return beta_end
    # Linear schedule:
    # progress = epoch / (n_epochs - 1)
    # return beta_start + progress * (beta_end - beta_start)

    progress = epoch / (n_epochs - 1)
    cosine = 0.5 * (1 - np.cos(np.pi * progress))
    return beta_start + cosine * (beta_end - beta_start)


def train_loop_vae(
    epoch: int,
    dataloader: DataLoader,
    model: BaseVAE,
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

        x_hat, mu, log_covariance = model(x)

        optimizer.zero_grad()
        loss = criterion(x_hat, x, mu, log_covariance)

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
    model: BaseVAE,
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

        x_hat, mu, log_covariance = model(x)
        loss = criterion(x_hat, x, mu, log_covariance)

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


def _append_component_history(
    history: dict[str, list[float]],
    component_values: dict[str, float],
):
    for key, value in component_values.items():
        if key not in history:
            raise ValueError(
                f"Component key '{key}' not found in history. Make sure to initialize the history dictionary with all expected keys before training."
                f"Current history keys: {list(history.keys())}, component keys: {list(component_values.keys())}"
            )
        history[key].append(float(value))


def export_loss_components(
    export_path: Path,
    train_components: dict[str, list[float]],
    test_components: dict[str, list[float]],
):
    all_keys = sorted(set(train_components.keys()) | set(test_components.keys()))
    if len(all_keys) == 0:
        return

    n_keys = len(all_keys)
    n_cols = 3
    n_rows = int(np.ceil(n_keys / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)

    for idx, key in enumerate(all_keys):
        ax = axes[idx // n_cols][idx % n_cols]
        if key in train_components:
            ax.plot(train_components[key], label="Train")
        if key in test_components:
            ax.plot(test_components[key], label="Test")
        pos_train = all(v > 0 for v in train_components.get(key, []))
        pos_test = all(v > 0 for v in test_components.get(key, []))
        if pos_train and pos_test:
            ax.set_yscale("log")
        ax.set_title(key)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.legend()

    for idx in range(n_keys, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(export_path / "loss_components_vae.png")
    plt.close(fig)


def export_results_vae(
    export_path: Path,
    model: BaseVAE,
    loss_list_train: list[float],
    loss_list_test: list[float],
    component_history_train: dict[str, list[float]] | None = None,
    component_history_test: dict[str, list[float]] | None = None,
) -> None:
    loss_list_train = np.array(loss_list_train)
    loss_list_test = np.array(loss_list_test)
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig, axes = plt.subplot_mosaic(
        [["A", "B"], ["C", "C"]],
        figsize=(12, 8),
    )
    ax1, ax2 = axes["A"], axes["B"]
    ax3 = axes["C"]
    ax1.plot(loss_list_train)
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.plot(loss_list_test)
    ax2.set_title("Test Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax3.plot(loss_list_train, label="Train Loss")
    ax3.plot(loss_list_test, label="Test Loss")
    ax3.set_title("Train vs Test Loss")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Loss")
    ax3.legend()

    if (loss_list_train > 0).all() and (loss_list_test > 0).all():
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax3.set_yscale("log")

    plt.tight_layout()
    plt.savefig(export_path / "loss_vae.png")
    plt.close(fig)

    np.save(export_path / "loss_train_vae.npy", loss_list_train)
    np.save(export_path / "loss_test_vae.npy", loss_list_test)

    if component_history_train is not None and component_history_test is not None:
        np.save(export_path / "loss_components_train_vae.npy", component_history_train)
        np.save(export_path / "loss_components_test_vae.npy", component_history_test)
        export_loss_components(export_path, component_history_train, component_history_test)

    torch.save(model.encoder.state_dict(), export_path / "encoder.pth")
    torch.save(model.decoder.state_dict(), export_path / "decoder.pth")


def visualize_reconstruction(
    model: BaseVAE,
    dataloader: DataLoader,
    device: torch.device,
    export_path: Path,
    n_samples: int = 10,
    img_shape: tuple[int, int, int] = (1, 28, 28),
    filename: str = "reconstruction_vae.png",
) -> None:
    img_reco_, img_true_ = [], []
    model.eval()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        x_reco, _, _ = model(x)

        img_reco_.append(x_reco.cpu().detach().numpy())
        img_true_.append(x.cpu().detach().numpy())
        img_reco = np.concatenate(img_reco_, axis=0)
        img_true = np.concatenate(img_true_, axis=0)
        if len(img_reco) >= n_samples:
            break

    indices = np.random.choice(len(img_reco), n_samples, replace=False)
    img_reco = img_reco[indices].reshape(n_samples, *img_shape)
    img_true = img_true[indices].reshape(n_samples, *img_shape)
    if len(img_shape) == 3:
        # Reshape from (n,c,h,w) to (n,h,w,c) for visualization
        img_reco = rearrange(img_reco, "n c h w -> n h w c")
        img_true = rearrange(img_true, "n c h w -> n h w c")
    elif len(img_shape) == 4:
        # Reshape from (n,c,h,w,d) to (n,d,h,w,c) for visualization
        img_reco = rearrange(img_reco, "n c h w d -> n d h w c")
        img_true = rearrange(img_true, "n c h w d -> n d h w c")
    cmap = "gray" if img_shape[0] == 1 else None

    fig, axes = plt.subplots(nrows=2, ncols=n_samples, figsize=(20, 4))
    for i in range(n_samples):
        if len(img_shape) == 3:
            axes[0, i].imshow(img_true[i].squeeze(), cmap=cmap)
            axes[1, i].imshow(img_reco[i].squeeze(), cmap=cmap)
        elif len(img_shape) == 4:
            d_mid = img_true.shape[1] // 2
            axes[0, i].imshow(img_true[i, d_mid].squeeze(), cmap=cmap)
            axes[1, i].imshow(img_reco[i, d_mid].squeeze(), cmap=cmap)

        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("True Images", fontsize=16)
    axes[1, 0].set_ylabel("Reconstructed Images", fontsize=16)
    plt.tight_layout()
    plt.savefig(export_path / filename)
    plt.close(fig)


def vae_trainer(vae_config: VAEConfig) -> nn.Module:
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
