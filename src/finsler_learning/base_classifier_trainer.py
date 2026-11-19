# Base trainer for classification in the latent space,
# using a pre-trained encoder and a classifier.


import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from .losses import NaNLossException
from .models import BaseEncoder


@dataclass
class ClassifierConfig:
    n_epochs: int
    dataloader: DataLoader
    dataloader_test: DataLoader
    encoder: BaseEncoder
    classifier: nn.Module
    optimizer: optim.Optimizer
    criterion: callable
    export_path: Path
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_progress_bar: bool = False
    n_classes: int = 2


def train_loop_classifier(
    epoch: int,
    dataloader: DataLoader,
    encoder: BaseEncoder,
    classifier: nn.Module,
    optimizer: optim.Optimizer,
    criterion: callable,
    device: torch.device,
    epoch_progress_bar: bool = False,
    num_classes: int = 2,
) -> tuple[float, float]:

    encoder.eval()
    classifier.train()

    metric = MulticlassAccuracy(num_classes=num_classes).to(device)

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
        logits = classifier(z)
        loss = criterion(logits, y)

        if torch.isnan(loss).any():
            raise NaNLossException(f"NaN loss encountered at epoch {epoch}, batch {i}. ")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.update(logits, y)
        loss_list_train_epoch.append(loss.item())

    accuracy_list_train_epoch = metric.compute().item()
    metric.reset()
    loss_train_epoch = np.mean(loss_list_train_epoch)
    return loss_train_epoch, accuracy_list_train_epoch


def test_loop_classifier(
    epoch: int,
    dataloader: DataLoader,
    encoder: BaseEncoder,
    classifier: nn.Module,
    criterion: callable,
    device: torch.device,
    num_classes: int = 2,
) -> tuple[float, float]:
    encoder.eval()
    classifier.eval()

    metric = MulticlassAccuracy(num_classes=num_classes).to(device)

    loss_list_test_epoch = []

    with torch.no_grad():
        for data in dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)

            z, log_var = encoder(x)
            y_pred = classifier(z)
            loss = criterion(y_pred, y)

            metric.update(y_pred, y)
            loss_list_test_epoch.append(loss.item())

    accuracy_test_epoch = metric.compute().item()
    loss_test_epoch = np.mean(loss_list_test_epoch)
    return loss_test_epoch, accuracy_test_epoch


def export_results(
    export_path: Path,
    classifier: nn.Module,
    loss_train: list[float],
    accuracy_train: list[float],
    loss_test: list[float],
    accuracy_test: list[float],
) -> None:
    loss_train = np.array(loss_train)
    accuracy_train = np.array(accuracy_train)
    loss_test = np.array(loss_test)
    accuracy_test = np.array(accuracy_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(loss_train, label="Train Loss")
    ax1.plot(loss_test, label="Test Loss")
    ax1.set_yscale("log")
    ax1.set_title("Loss")
    ax1.set_ylabel("Loss")
    ax2.plot(accuracy_train, label="Train Accuracy")
    ax2.plot(accuracy_test, label="Test Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_ylabel("Accuracy")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(export_path / "loss_acc_latent_classifier.png")
    plt.close(fig)

    np.save(export_path / "loss_train_latent_classifier.npy", loss_train)
    np.save(export_path / "accuracy_train_latent_classifier.npy", accuracy_train)
    np.save(export_path / "loss_test_latent_classifier.npy", loss_test)
    np.save(export_path / "accuracy_test_latent_classifier.npy", accuracy_test)

    torch.save(classifier.state_dict(), export_path / "latent_classifier.pth")


def classifier_trainer(
    classifier_config: ClassifierConfig,
) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    export_path = Path(classifier_config.export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    loss_train, accuracy_train = [], []
    loss_test, accuracy_test = [], []

    classifier_config.encoder.to(classifier_config.device)
    classifier_config.classifier.to(classifier_config.device)

    pbar = tqdm(range(classifier_config.n_epochs), desc="Training Classifier", unit="epoch")
    for epoch in pbar:
        loss_epoch_train, accuracy_epoch_train = train_loop_classifier(
            epoch=epoch,
            dataloader=classifier_config.dataloader,
            encoder=classifier_config.encoder,
            classifier=classifier_config.classifier,
            optimizer=classifier_config.optimizer,
            criterion=classifier_config.criterion,
            device=classifier_config.device,
            epoch_progress_bar=classifier_config.epoch_progress_bar,
            num_classes=classifier_config.n_classes,
        )
        loss_epoch_test, accuracy_epoch_test = test_loop_classifier(
            epoch=epoch,
            dataloader=classifier_config.dataloader_test,
            encoder=classifier_config.encoder,
            classifier=classifier_config.classifier,
            criterion=classifier_config.criterion,
            device=classifier_config.device,
            num_classes=classifier_config.n_classes,
        )
        loss_train.append(loss_epoch_train)
        accuracy_train.append(accuracy_epoch_train)
        loss_test.append(loss_epoch_test)
        accuracy_test.append(accuracy_epoch_test)

        pbar.set_postfix(
            {
                "loss_train": f"{loss_epoch_train:.4f}",
                "accuracy_train": f"{accuracy_epoch_train:.4f}",
                "loss_test": f"{loss_epoch_test:.4f}",
                "accuracy_test": f"{accuracy_epoch_test:.4f}",
            }
        )
        export_results(
            export_path=export_path,
            classifier=classifier_config.classifier,
            loss_train=loss_train,
            accuracy_train=accuracy_train,
            loss_test=loss_test,
            accuracy_test=accuracy_test,
        )

    logger = logging.getLogger()
    logger.info(f"Classifier training finished. Results saved to {export_path}")
    logger.info(
        f"Final train loss: {loss_train[-1]:.4f}, train accuracy: {accuracy_train[-1]:.4f}"
    )
    logger.info(
        f"Final test loss: {loss_test[-1]:.4f}, test accuracy: {accuracy_test[-1]:.4f}"
    )
    return classifier_config.classifier, loss_train, accuracy_train, loss_test, accuracy_test
