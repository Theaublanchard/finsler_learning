# Learn an hyperplane model for the Randers metric
import torch
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from geodesic_toolbox import CoMetric, IdentityCoMetric
from .losses import GroupedMarginLoss
from .models import BaseEncoder, Omega


@dataclass
class RandersLearningConfig:
    """
    Configuration for Randers metric learning.
    """

    dataloader_train: DataLoader
    dataloader_test: DataLoader
    encoder: BaseEncoder
    omega: Omega
    loss_fn: GroupedMarginLoss
    n_epochs: int
    export_path: Path
    cometric: CoMetric = IdentityCoMetric()
    lr: float = 1e-3
    weight_decay: float = 1e-6
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    show_pbar: bool = True


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    """Freeze a module so it can be used as a fixed feature provider."""
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def train_loop_hyperplane(
    dataloader: DataLoader,
    encoder: BaseEncoder,
    omega: Omega,
    loss_dir: GroupedMarginLoss,
    loss_amp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cometric: CoMetric = IdentityCoMetric(),
    show_pbar: bool = True,
) -> tuple[float, float]:
    omega.train()
    encoder.eval()
    loss_list_train_epoch_dir = []
    loss_list_train_epoch_amp = []
    if show_pbar:
        pbar = tqdm(dataloader, desc="Train epoch", unit="batch", total=len(dataloader),leave=False)
    else:
        pbar = dataloader
    # for i, data in enumerate(dataloader):
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        x, t = data
        x = x.to(device)
        t = t.to(device)
        with torch.no_grad():
            z, log_var = encoder(x)
        z = z.detach()

        w = omega.direction(z)
        loss_direction = loss_dir(w, z, t, cometric=cometric)
        with torch.no_grad():
            true_accuracy = loss_dir.acc.detach().clone().unsqueeze(1)
            true_val = 2 * (true_accuracy - 0.5).relu()
        rho = omega.amplitude(z)
        loss_acc = loss_amp(rho, true_val)

        optimizer.zero_grad()
        (loss_direction + loss_acc).backward()
        optimizer.step()

        loss_list_train_epoch_dir.append(loss_direction.item())
        loss_list_train_epoch_amp.append(loss_acc.item())
    loss_train_epoch_dir = np.mean(loss_list_train_epoch_dir)
    loss_train_epoch_amp = np.mean(loss_list_train_epoch_amp)
    return loss_train_epoch_dir, loss_train_epoch_amp


@torch.no_grad()
def test_loop_hyperplane(
    dataloader: DataLoader,
    encoder: BaseEncoder,
    omega: Omega,
    loss_dir: GroupedMarginLoss,
    loss_amp: torch.nn.Module,
    device: torch.device,
    cometric: CoMetric = IdentityCoMetric(),
    show_pbar: bool = True,
) -> tuple[float, float]:
    omega.eval()
    loss_list_test_epoch_dir = []
    loss_list_test_epoch_amp = []
    if show_pbar:
        pbar = tqdm(dataloader, desc="Test epoch", unit="batch", total=len(dataloader),leave=False)
    else:
        pbar = dataloader
    for i, data in enumerate(pbar):
        x, t = data
        x = x.to(device)
        t = t.to(device)
        z, log_var = encoder(x)

        w = omega.direction(z)
        loss_direction = loss_dir(w, z, t, cometric=cometric)
        true_accuracy = loss_dir.acc.detach().unsqueeze(1)
        true_val = 2 * (true_accuracy - 0.5).relu()
        rho = omega.amplitude(z)
        loss_acc = loss_amp(rho, true_val)

        loss_list_test_epoch_dir.append(loss_direction.item())
        loss_list_test_epoch_amp.append(loss_acc.item())
    loss_test_epoch_dir = np.mean(loss_list_test_epoch_dir)
    loss_test_epoch_amp = np.mean(loss_list_test_epoch_amp)
    return loss_test_epoch_dir, loss_test_epoch_amp


def export_hyperplane(
    hyperplane_model: torch.nn.Module,
    export_path: Path,
    train_loss_dir: list[float],
    test_loss_dir: list[float],
    train_loss_amp: list[float],
    test_loss_amp: list[float],
):
    train_loss_dir = np.array(train_loss_dir)
    test_loss_dir = np.array(test_loss_dir)
    train_loss_amp = np.array(train_loss_amp)
    test_loss_amp = np.array(test_loss_amp)

    torch.save(hyperplane_model.direction.state_dict(), export_path / "direction_model.pth")
    torch.save(hyperplane_model.amplitude.state_dict(), export_path / "amplitude_model.pth")

    np.save(export_path / "train_loss_dir.npy", train_loss_dir)
    np.save(export_path / "test_loss_dir.npy", test_loss_dir)
    np.save(export_path / "train_loss_amp.npy", train_loss_amp)
    np.save(export_path / "test_loss_amp.npy", test_loss_amp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_loss_dir, label="Train Loss Direction", color="blue")
    ax1.plot(test_loss_dir, label="Test Loss Direction", color="orange")
    ax1.set_title("Loss Direction Curves")
    ax1.set_ylabel("Loss Direction")
    ax2.plot(train_loss_amp, label="Train Loss Amplitude", color="blue")
    ax2.plot(test_loss_amp, label="Test Loss Amplitude", color="orange")
    ax2.set_title("Loss Amplitude Curves")
    ax2.set_ylabel("Loss Amplitude")

    if (train_loss_dir > 0).all() and (test_loss_dir > 0).all():
        ax1.set_yscale("log")
    if (train_loss_amp > 0).all() and (test_loss_amp > 0).all():
        ax2.set_yscale("log")

    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(export_path / "loss_curves_randers.png")
    plt.close(fig)


def loss_amplitude(rho, true_val):
    loss = torch.nn.functional.mse_loss(
        rho,
        true_val,
        # weight=torch.exp(-(true_val**4)),
    )
    return loss


def train_direction(config: RandersLearningConfig) -> tuple[Omega, list, list, list, list]:
    """
    Train a hyperplane model for the Randers metric.
    """
    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    encoder = freeze_module(config.encoder.to(config.device))
    omega = config.omega.to(config.device)
    if isinstance(config.cometric, torch.nn.Module):
        cometric = freeze_module(config.cometric.to(config.device))
    else:
        cometric = config.cometric
    optimizer = torch.optim.Adam(
        omega.direction.parameters(),
        weight_decay=config.weight_decay,
        lr=config.lr,
    )
    train_loss_dir_list, test_loss_dir_list = [], []
    train_loss_amp_list, test_loss_amp_list = [], []
    train_loss_dir, test_loss_dir = 0.0, 0.0
    train_loss_amp, test_loss_amp = 0.0, 0.0

    pbar = tqdm(
        range(config.n_epochs),
        desc="Training Randers Hyperplane",
        unit="epoch",
        total=config.n_epochs,
    )
    pbar.set_postfix(
        {
            "train_loss_dir": train_loss_dir,
            "test_loss_dir": test_loss_dir,
            "train_loss_amp": train_loss_amp,
            "test_loss_amp": test_loss_amp,
        }
    )
    for epoch in pbar:
        loss_train_epoch_dir, loss_train_epoch_amp = train_loop_hyperplane(
            dataloader=config.dataloader_train,
            encoder=encoder,
            omega=omega,
            loss_dir=config.loss_fn,
            loss_amp=loss_amplitude,
            optimizer=optimizer,
            device=config.device,
            cometric=cometric,
            show_pbar=config.show_pbar
        )
        pbar.set_postfix(
            {
                "train_loss_dir": loss_train_epoch_dir,
                "train_loss_amp": loss_train_epoch_amp,
                "test_loss_dir": test_loss_dir,
                "test_loss_amp": test_loss_amp,
            }
        )

        loss_test_epoch_dir, loss_test_epoch_amp = test_loop_hyperplane(
            dataloader=config.dataloader_test,
            encoder=encoder,
            omega=omega,
            loss_dir=config.loss_fn,
            loss_amp=loss_amplitude,
            device=config.device,
            cometric=cometric,
            show_pbar=config.show_pbar
        )
        pbar.set_postfix(
            {
                "train_loss_dir": loss_train_epoch_dir,
                "test_loss_dir": loss_test_epoch_dir,
                "train_loss_amp": loss_train_epoch_amp,
                "test_loss_amp": loss_test_epoch_amp,
            }
        )

        train_loss_dir_list.append(loss_train_epoch_dir)
        test_loss_dir_list.append(loss_test_epoch_dir)
        train_loss_amp_list.append(loss_train_epoch_amp)
        test_loss_amp_list.append(loss_test_epoch_amp)

        export_hyperplane(
            hyperplane_model=omega,
            export_path=export_path,
            train_loss_dir=train_loss_dir_list,
            test_loss_dir=test_loss_dir_list,
            train_loss_amp=train_loss_amp_list,
            test_loss_amp=test_loss_amp_list,
        )

    export_hyperplane(
        hyperplane_model=omega,
        export_path=export_path,
        train_loss_dir=train_loss_dir_list,
        test_loss_dir=test_loss_dir_list,
        train_loss_amp=train_loss_amp_list,
        test_loss_amp=test_loss_amp_list,
    )
    return (
        omega,
        train_loss_dir_list,
        test_loss_dir_list,
        train_loss_amp_list,
        test_loss_amp_list,
    )


def train_amplitude(
    config: RandersLearningConfig,
    train_loss_dir_list: list[float],
    test_loss_dir_list: list[float],
) -> tuple[Omega, list, list, list, list]:
    """
    Train a hyperplane model for the Randers metric.
    """
    export_path = config.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    encoder = freeze_module(config.encoder.to(config.device))
    omega = config.omega.to(config.device)
    if isinstance(config.cometric, torch.nn.Module):
        cometric = freeze_module(config.cometric.to(config.device))
    else:
        cometric = config.cometric
    optimizer = torch.optim.Adam(
        [
            {"params": omega.amplitude.parameters(), "lr": config.lr / 2},
        ],
        weight_decay=config.weight_decay,
        lr=config.lr,
    )
    train_loss_amp_list, test_loss_amp_list = [], []
    train_loss_amp, test_loss_amp = 0.0, 0.0

    pbar = tqdm(
        range(config.n_epochs),
        desc="Training Randers Hyperplane",
        unit="epoch",
        total=config.n_epochs,
    )
    pbar.set_postfix(
        {
            "train_loss_dir": train_loss_dir_list[-1],
            "test_loss_dir": test_loss_dir_list[-1],
            "train_loss_amp": train_loss_amp,
            "test_loss_amp": test_loss_amp,
        }
    )
    for epoch in pbar:
        _, loss_train_epoch_amp = train_loop_hyperplane(
            dataloader=config.dataloader_train,
            encoder=encoder,
            omega=omega,
            loss_dir=config.loss_fn,
            loss_amp=loss_amplitude,
            optimizer=optimizer,
            device=config.device,
            cometric=cometric,
            show_pbar=config.show_pbar
        )
        pbar.set_postfix(
            {
                "train_loss_dir": train_loss_dir_list[-1],
                "train_loss_amp": loss_train_epoch_amp,
                "test_loss_dir": test_loss_dir_list[-1],
                "test_loss_amp": test_loss_amp,
            }
        )

        _, loss_test_epoch_amp = test_loop_hyperplane(
            dataloader=config.dataloader_test,
            encoder=encoder,
            omega=omega,
            loss_dir=config.loss_fn,
            loss_amp=loss_amplitude,
            device=config.device,
            cometric=cometric,
            show_pbar=config.show_pbar
        )
        pbar.set_postfix(
            {
                "train_loss_dir": train_loss_dir_list[-1],
                "test_loss_dir": test_loss_dir_list[-1],
                "train_loss_amp": loss_train_epoch_amp,
                "test_loss_amp": loss_test_epoch_amp,
            }
        )

        train_loss_amp_list.append(loss_train_epoch_amp)
        test_loss_amp_list.append(loss_test_epoch_amp)

        export_hyperplane(
            hyperplane_model=omega,
            export_path=export_path,
            train_loss_dir=train_loss_dir_list,
            test_loss_dir=test_loss_dir_list,
            train_loss_amp=train_loss_amp_list,
            test_loss_amp=test_loss_amp_list,
        )

    export_hyperplane(
        hyperplane_model=omega,
        export_path=export_path,
        train_loss_dir=train_loss_dir_list,
        test_loss_dir=test_loss_dir_list,
        train_loss_amp=train_loss_amp_list,
        test_loss_amp=test_loss_amp_list,
    )
    return (
        omega,
        train_loss_dir_list,
        test_loss_dir_list,
        train_loss_amp_list,
        test_loss_amp_list,
    )


@torch.no_grad()
def get_accuracy_hyperplane(dataloader: DataLoader, config: RandersLearningConfig) -> float:
    """
    Get the accuracy of the hyperplane model for the Randers metric.
    """
    acc = []
    for data in dataloader:
        x, t = data
        x = x.to(config.device)
        t = t.to(config.device)
        z, _ = config.encoder(x)
        z = z.detach()

        w = config.omega.direction(z)
        _ = config.loss_fn(w, z, t)
        true_accuracy = config.loss_fn.acc.detach().clone()
        acc.append(true_accuracy.cpu())
    acc = torch.cat(acc, dim=0)
    return acc


def export_accuracy_hyperplane(
    acc_train: torch.Tensor, acc_test: torch.Tensor, export_path: Path
) -> None:
    """
    Export the accuracy of the hyperplane model for the Randers metric.
    """
    export_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(
        acc_train.numpy(),
        color="blue",
        label="Train",
        ax=ax[0],
        bins=20,
    )
    sns.histplot(
        acc_test.numpy(),
        color="orange",
        label="Test",
        ax=ax[1],
        bins=20,
    )
    for axis in ax:
        axis.set_xlabel("Accuracy")
        axis.set_ylabel("Count")
        axis.legend()
    plt.suptitle("Histogram of local hyperplane accuracy")
    plt.tight_layout()
    plt.savefig(export_path / "accuracy_histogram.png")
    plt.close(fig)


def assess_hyperplane_precision(config: RandersLearningConfig) -> None:
    """
    Get the precision of the hyperplane model for the Randers metric.
    """
    acc_train = get_accuracy_hyperplane(config.dataloader_train, config)
    acc_test = get_accuracy_hyperplane(config.dataloader_test, config)

    logger = logging.getLogger()
    logger.info(f"Train accuracy: {acc_train.mean().item()} +/- {acc_train.std().item()}")
    logger.info(f"Test accuracy: {acc_test.mean().item()} +/- {acc_test.std().item()}")
    export_accuracy_hyperplane(
        acc_train=acc_train, acc_test=acc_test, export_path=config.export_path
    )


def train_hyperplane(config: RandersLearningConfig) -> tuple[Omega, list, list, list, list]:
    """
    Train a hyperplane model for the Randers metric.
    """
    _, train_loss_dir_list, test_loss_dir_list, _, _ = train_direction(config=config)
    train_amplitude(
        config=config,
        train_loss_dir_list=train_loss_dir_list,
        test_loss_dir_list=test_loss_dir_list,
    )
    assess_hyperplane_precision(config=config)
