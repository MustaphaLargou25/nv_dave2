"""Configuration objects for the DAVE-2 behavioral cloning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    """Settings for raw data ingestion and train/validation splitting."""

    raw_data_dir: Path
    steering_correction: float = 0.2
    test_size: float = 0.2
    random_state: int = 42

    @property
    def csv_path(self) -> Path:
        return self.raw_data_dir / "driving_log.csv"

    @property
    def image_dir(self) -> Path:
        return self.raw_data_dir / "IMG"


@dataclass(slots=True)
class PreprocessingConfig:
    """Image preprocessing settings used by the dataset."""

    crop_top: int = 60
    crop_bottom: int = 25
    resize_height: int = 66
    resize_width: int = 200


@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters for DAVE-2."""

    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-4
    num_workers: int = 0
    pin_memory: bool = True
    checkpoint_path: Path = Path("artifacts/best_dave2.pth")


@dataclass(slots=True)
class PerceptionConfig:
    """Settings for the SAM-style perception stack."""

    image_size: tuple[int, int] = (256, 512)
    backbone_dims: tuple[int, int, int] = (64, 128, 256)
    embedding_dim: int = 256
    scene_token_grid: tuple[int, int] = (4, 8)
    num_segmentation_classes: int = 8
    num_object_classes: int = 8
    num_detection_queries: int = 32
    lane_anchor_count: int = 24
    track_embedding_dim: int = 64
    max_tracks: int = 48
    track_score_threshold: float = 0.35
    track_match_threshold: float = 0.30
    road_threshold: float = 0.50
    lane_threshold: float = 0.50


@dataclass(slots=True)
class FusionConfig:
    """Settings that govern perception-policy fusion experiments."""

    strategy: str = "multi_branch"
    fusion_dim: int = 256
    num_heads: int = 8
    depth: int = 2
    dropout: float = 0.10


@dataclass(slots=True)
class PolicyConfig:
    """Action-head and optional uncertainty settings."""

    action_dim: int = 3
    hidden_dim: int = 256
    dropout: float = 0.10
    ego_state_dim: int = 6
    predict_uncertainty: bool = True


@dataclass(slots=True)
class HybridModelConfig:
    """Top-level configuration for the hybrid perception plus policy model."""

    vit_variant: str = "mini"
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
