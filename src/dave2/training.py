"""Phase 4: device-aware DAVE-2 training and evaluation utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(slots=True)
class EpochMetrics:
    """Scalar training metrics captured at the end of each epoch."""

    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(slots=True)
class TrainingResult:
    """Final artifact path and per-epoch training history."""

    best_validation_loss: float
    checkpoint_path: Path
    history: list[EpochMetrics]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Choose GPU when available and otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one full pass over a dataloader and return mean loss."""

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0

    context_manager = torch.enable_grad if is_training else torch.no_grad
    with context_manager():
        for images, steering_angles in dataloader:
            images = images.to(device, non_blocking=True)
            steering_angles = steering_angles.to(device, non_blocking=True)

            predictions = model(images)
            loss = loss_function(predictions, steering_angles)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Dataloader produced zero examples.")

    return total_loss / total_examples


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
    device: torch.device,
) -> TrainingResult:
    """Train DAVE-2 and save the weights with the best validation loss."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    best_validation_loss = float("inf")
    history: list[EpochMetrics] = []

    for epoch_index in range(1, epochs + 1):
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )
        validation_loss = run_epoch(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
            device=device,
            optimizer=None,
        )

        history.append(
            EpochMetrics(
                epoch=epoch_index,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )
        )

        print(
            f"Epoch {epoch_index:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), checkpoint_path)

    return TrainingResult(
        best_validation_loss=best_validation_loss,
        checkpoint_path=checkpoint_path,
        history=history,
    )
