"""Phase 2: preprocessing and PyTorch dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import PreprocessingConfig


def crop_to_road(image_rgb: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Remove sky and vehicle hood while preserving the road region."""

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    bottom_index = image_rgb.shape[0] - config.crop_bottom
    if config.crop_top >= bottom_index:
        raise ValueError("Crop settings remove the full image.")

    return image_rgb[config.crop_top:bottom_index, :, :]


def preprocess_image(
    image_rgb: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Apply the DAVE-2 preprocessing sequence and return `float32` HWC data."""

    cropped = crop_to_road(image_rgb, config)
    resized = cv2.resize(
        cropped,
        (config.resize_width, config.resize_height),
        interpolation=cv2.INTER_AREA,
    )
    image_yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
    normalized = image_yuv.astype(np.float32) / 127.5 - 1.0
    return normalized


@dataclass(slots=True)
class Sample:
    """Simple typed container that describes a processed sample."""

    image: torch.Tensor
    steering: torch.Tensor


class BehavioralCloningDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for NVIDIA DAVE-2 behavioral cloning."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessing: PreprocessingConfig | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.preprocessing = preprocessing or PreprocessingConfig()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image_path = row["image_path"]

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"OpenCV could not read image '{image_path}'.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        processed = preprocess_image(image_rgb, self.preprocessing)
        image_tensor = torch.from_numpy(np.transpose(processed, (2, 0, 1)))
        steering_tensor = torch.tensor([row["steering"]], dtype=torch.float32)
        return image_tensor, steering_tensor


def build_dataloaders(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    preprocessing: PreprocessingConfig | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create shuffled training and deterministic validation dataloaders."""

    preprocessing = preprocessing or PreprocessingConfig()
    train_dataset = BehavioralCloningDataset(train_frame, preprocessing=preprocessing)
    validation_dataset = BehavioralCloningDataset(
        validation_frame,
        preprocessing=preprocessing,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader

