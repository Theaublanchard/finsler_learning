# Base trainer for regression in the latent space,
# using a pre-trained encoder and a latent regressor.
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError
from tqdm import tqdm

from .losses import NaNLossException
from .models import BaseEncoder


@dataclass
class LatentRegressorConfig:
    n_epochs: int
    dataloader: DataLoader
    dataloader_test: DataLoader
    encoder: BaseEncoder
    regressor: nn.Module
    optimizer: optim.Optimizer
    criterion: callable
    export_path: Path
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_progress_bar: bool = False


def train_loop_latent_regressor(
    epoch: int,
    dataloader: DataLoader,
    encoder: BaseEncoder,
    regressor: nn.Module,
    optimizer: optim.Optimizer,
    criterion: callable,
    device: torch.device,
    epoch_progress_bar: bool = False,
) -> tuple[float, float]:

    encoder.eval()
    regressor.train()

    metric = MeanAbsolutePercentageError().to(device)

    loss_list_train_epoch = []
    if epoch_progress_bar:
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}",
            unit="batch",
        )
    else:
        pbar = enumerate(dataloader)

    for i, data in pbar:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        z, log_var = encoder(x)

        logits = regressor(z).squeeze(dim=1)
        loss = criterion(logits, y.to(logits.dtype))

        if torch.isnan(loss).any():
            raise NaNLossException(f"NaN loss encountered at epoch {epoch}, batch {i}. ")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.update(logits, y)
        loss_list_train_epoch.append(loss.item())

    mape_train_epoch = metric.compute().item()
    metric.reset()
    loss_train_epoch = np.mean(loss_list_train_epoch)
    return loss_train_epoch, mape_train_epoch


def test_loop_latent_regressor(
    epoch: int,
    dataloader: DataLoader,
    encoder: nn.Module,
    regressor: nn.Module,
    criterion: callable,
    device: torch.device,
) -> tuple[float, float]:
    encoder.eval()
    regressor.eval()

    metric = MeanAbsolutePercentageError().to(device)

    loss_list_test_epoch = []

    with torch.no_grad():
        for data in dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            z, log_var = encoder(x)

            y_pred = regressor(z).squeeze(dim=1)
            loss = criterion(y_pred, y.to(y_pred.dtype))

            metric.update(y_pred, y)
            loss_list_test_epoch.append(loss.item())

    mape_test_epoch = metric.compute().item()
    loss_test_epoch = np.mean(loss_list_test_epoch)
    return loss_test_epoch, mape_test_epoch


def export_results(
    export_path: Path,
    regressor: nn.Module,
    loss_train: list[float],
    mape_train: list[float],
    loss_test: list[float],
    mape_test: list[float],
) -> None:
    loss_train = np.array(loss_train)
    mape_train = np.array(mape_train)
    loss_test = np.array(loss_test)
    mape_test = np.array(mape_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(loss_train, label="Train Loss")
    ax1.plot(loss_test, label="Test Loss")
    ax1.set_yscale("log")
    ax1.set_title("Loss")
    ax1.set_ylabel("Loss")
    ax2.plot(mape_train, label="Train MAPE")
    ax2.plot(mape_test, label="Test MAPE")
    ax2.set_title("Mean Absolute Percentage Error (MAPE)")
    ax2.set_ylabel("MAPE")
    ax2.set_ylim(0, 1.2)
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(export_path / "loss_mape_latent_regressor.png")
    plt.close(fig)

    np.save(export_path / "loss_train_latent_regressor.npy", loss_train)
    np.save(export_path / "mape_train_latent_regressor.npy", mape_train)
    np.save(export_path / "loss_test_latent_regressor.npy", loss_test)
    np.save(export_path / "mape_test_latent_regressor.npy", mape_test)

    torch.save(regressor.state_dict(), export_path / "latent_regressor.pth")


def latent_regressor_trainer(
    regressor_config: LatentRegressorConfig,
) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    export_path = Path(regressor_config.export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    loss_train, mape_train = [], []
    loss_test, mape_test = [], []

    regressor_config.encoder.to(regressor_config.device)
    regressor_config.regressor.to(regressor_config.device)

    pbar = tqdm(
        range(regressor_config.n_epochs), desc="Training Latent Regressor", unit="epoch"
    )
    for epoch in pbar:
        loss_epoch_train, mape_epoch_train = train_loop_latent_regressor(
            epoch=epoch,
            dataloader=regressor_config.dataloader,
            encoder=regressor_config.encoder,
            regressor=regressor_config.regressor,
            optimizer=regressor_config.optimizer,
            criterion=regressor_config.criterion,
            device=regressor_config.device,
            epoch_progress_bar=regressor_config.epoch_progress_bar,
        )
        loss_epoch_test, mape_epoch_test = test_loop_latent_regressor(
            epoch=epoch,
            dataloader=regressor_config.dataloader_test,
            encoder=regressor_config.encoder,
            regressor=regressor_config.regressor,
            criterion=regressor_config.criterion,
            device=regressor_config.device,
        )
        loss_train.append(loss_epoch_train)
        mape_train.append(mape_epoch_train)
        loss_test.append(loss_epoch_test)
        mape_test.append(mape_epoch_test)

        pbar.set_postfix(
            {
                "loss_train": f"{loss_epoch_train:.4f}",
                "mape_train": f"{mape_epoch_train:.4f}",
                "loss_test": f"{loss_epoch_test:.4f}",
                "mape_test": f"{mape_epoch_test:.4f}",
            }
        )
        export_results(
            export_path=export_path,
            regressor=regressor_config.regressor,
            loss_train=loss_train[1:],
            mape_train=mape_train[1:],  # Skip first epoch for better visualization
            loss_test=loss_test,
            mape_test=mape_test,
        )

    logger = logging.getLogger()
    logger.info(f"Latent regressor training finished. Results saved to {export_path}")
    logger.info(f"Final train loss: {loss_train[-1]:.4f}, train MAPE: {mape_train[-1]:.4f}")
    logger.info(f"Final test loss: {loss_test[-1]:.4f}, test MAPE: {mape_test[-1]:.4f}")
    return regressor_config.regressor, loss_train, mape_train, loss_test, mape_test
