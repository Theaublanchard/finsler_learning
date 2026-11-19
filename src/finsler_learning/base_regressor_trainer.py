# Base trainer for regression from the image space.

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


@dataclass
class RegressorConfig:
    n_epochs: int
    dataloader: DataLoader
    dataloader_test: DataLoader
    regressor: nn.Module
    optimizer: optim.Optimizer
    criterion: callable
    export_path: Path
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_progress_bar: bool = False
    scheduler: optim.lr_scheduler.LRScheduler | None = None


def train_loop_regressor(
    epoch: int,
    dataloader: DataLoader,
    regressor: nn.Module,
    optimizer: optim.Optimizer,
    criterion: callable,
    device: torch.device,
    epoch_progress_bar: bool = False,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float]:
    regressor.train()

    # metric = MeanSquaredError().to(device)
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

        y_hat = regressor(x).squeeze(dim=1)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.update(y_hat, y)
        loss_list_train_epoch.append(loss.item())

    if scheduler is not None:
        scheduler.step()

    metric_list_train_epoch = metric.compute().item()
    metric.reset()
    loss_train_epoch = np.mean(loss_list_train_epoch)
    return loss_train_epoch, metric_list_train_epoch


def test_loop_classifier(
    epoch: int,
    dataloader: DataLoader,
    regressor: nn.Module,
    criterion: callable,
    device: torch.device,
) -> tuple[float, float]:
    regressor.eval()

    # metric = MeanSquaredError().to(device)
    metric = MeanAbsolutePercentageError().to(device)

    loss_list_test_epoch = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            y_hat = regressor(x).squeeze(dim=1)
            loss = criterion(y_hat, y)

            metric.update(y_hat, y)
            loss_list_test_epoch.append(loss.item())

    metric_list_test_epoch = metric.compute().item()
    metric.reset()
    loss_test_epoch = np.mean(loss_list_test_epoch)
    return loss_test_epoch, metric_list_test_epoch


def export_results(
    export_path: Path,
    regressor: nn.Module,
    loss_train: list[float],
    metric_train: list[float],
    loss_test: list[float],
    metric_test: list[float],
) -> None:
    loss_train = np.array(loss_train)
    metric_train = np.array(metric_train)
    loss_test = np.array(loss_test)
    metric_test = np.array(metric_test)

    # Export plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(loss_train, label="Train Loss")
    ax1.plot(loss_test, label="Test Loss")
    ax1.set_yscale("log")
    ax1.set_title("Loss")
    ax1.set_ylabel("Loss")
    ax2.plot(metric_train, label="Train metric")
    ax2.plot(metric_test, label="Test metric")
    ax2.set_title("metric")
    ax2.set_ylabel("metric")
    ax2.set_ylim(0, 1.1)
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(export_path / "loss_metric.png")
    plt.close(fig)

    # Export model and logs
    np.save(export_path / "loss_train_regressor.npy", loss_train)
    np.save(export_path / "metric_train_regressor.npy", metric_train)
    np.save(export_path / "loss_test_regressor.npy", loss_test)
    np.save(export_path / "metric_test_regressor.npy", metric_test)

    torch.save(regressor.state_dict(), export_path / "regressor.pth")


def regressor_trainer(
    regressor_config: RegressorConfig,
) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    export_path = Path(regressor_config.export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    loss_train, metric_train = [], []
    loss_test, metric_test = [], []

    regressor_config.regressor.to(regressor_config.device)

    pbar = tqdm(range(regressor_config.n_epochs), desc="Training Regressor", unit="epoch")
    for epoch in pbar:
        loss_epoch_train, metric_epoch_train = train_loop_regressor(
            epoch=epoch,
            dataloader=regressor_config.dataloader,
            regressor=regressor_config.regressor,
            optimizer=regressor_config.optimizer,
            criterion=regressor_config.criterion,
            device=regressor_config.device,
            epoch_progress_bar=regressor_config.epoch_progress_bar,
            scheduler=regressor_config.scheduler,
        )
        loss_epoch_test, metric_epoch_test = test_loop_classifier(
            epoch=epoch,
            dataloader=regressor_config.dataloader_test,
            regressor=regressor_config.regressor,
            criterion=regressor_config.criterion,
            device=regressor_config.device,
        )

        loss_train.append(loss_epoch_train)
        metric_train.append(metric_epoch_train)
        loss_test.append(loss_epoch_test)
        metric_test.append(metric_epoch_test)

        pbar.set_postfix(
            {
                "loss_train": f"{loss_epoch_train:.4f}",
                "metric_train": f"{metric_epoch_train:.4f}",
                "loss_test": f"{loss_epoch_test:.4f}",
                "metric_test": f"{metric_epoch_test:.4f}",
            }
        )

        export_results(
            export_path=export_path,
            regressor=regressor_config.regressor,
            loss_train=loss_train,
            metric_train=metric_train,
            loss_test=loss_test,
            metric_test=metric_test,
        )
    logger = logging.getLogger()
    logger.info(f"Regressor training completed. Results saved to {export_path}")
    logger.info(
        f"Final train loss: {loss_train[-1]:.4f}, Final test loss: {loss_test[-1]:.4f}"
    )
    logger.info(
        f"Final train metric: {metric_train[-1]:.4f}, Final test metric: {metric_test[-1]:.4f}"
    )
    return regressor_config.regressor, loss_train, metric_train, loss_test, metric_test
