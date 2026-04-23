"""NVIDIA DAVE-2 behavioral cloning package."""

from .config import (
    DataConfig,
    FusionConfig,
    HybridModelConfig,
    PerceptionConfig,
    PolicyConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from .data import (
    build_behavioral_cloning_dataframe,
    save_dataset_split_manifests,
    split_behavioral_cloning_dataframe,
    summarize_behavioral_cloning_dataframe,
)
from .dataset import BehavioralCloningDataset, build_dataloaders, preprocess_image
from .fusion import HybridFusionModule
from .hybrid_model import DrivingPolicyOutput, HybridDrivingModel, build_hybrid_model
from .model import (
    DAVE2,
    DAVE2_VIT_VARIANTS,
    DAVE2VisionEncoder,
    build_model,
    count_parameters,
    describe_variant,
    get_dave2_vit_config,
)
from .perception import (
    ControlFeatures,
    DetectionOutput,
    LaneDetectionModule,
    LaneOutput,
    PerceptionOutput,
    RoadOutput,
    SAMPerceptionWrapper,
    SegmentationOutput,
    TrackMemory,
    TrackingModule,
    TrackingOutput,
)
from .training import fit, select_device

__all__ = [
    "BehavioralCloningDataset",
    "ControlFeatures",
    "DAVE2",
    "DAVE2_VIT_VARIANTS",
    "DAVE2VisionEncoder",
    "DataConfig",
    "DetectionOutput",
    "DrivingPolicyOutput",
    "FusionConfig",
    "HybridDrivingModel",
    "HybridFusionModule",
    "HybridModelConfig",
    "LaneDetectionModule",
    "LaneOutput",
    "PerceptionConfig",
    "PerceptionOutput",
    "PolicyConfig",
    "PreprocessingConfig",
    "RoadOutput",
    "SAMPerceptionWrapper",
    "SegmentationOutput",
    "TrackMemory",
    "TrackingModule",
    "TrackingOutput",
    "TrainingConfig",
    "build_behavioral_cloning_dataframe",
    "build_dataloaders",
    "build_hybrid_model",
    "build_model",
    "count_parameters",
    "describe_variant",
    "fit",
    "get_dave2_vit_config",
    "preprocess_image",
    "save_dataset_split_manifests",
    "select_device",
    "split_behavioral_cloning_dataframe",
    "summarize_behavioral_cloning_dataframe",
]
