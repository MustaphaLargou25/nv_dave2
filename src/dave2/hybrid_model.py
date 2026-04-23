"""Hybrid SAM-style perception plus ViT-DAVE2 driving model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import HybridModelConfig, PolicyConfig
from .fusion import HybridFusionModule
from .model import DAVE2VisionEncoder
from .perception import PerceptionOutput, SAMPerceptionWrapper, TrackMemory


@dataclass(slots=True)
class DrivingPolicyOutput:
    actions: Tensor
    action_uncertainty: Tensor | None
    fused_state: Tensor
    perception: PerceptionOutput


class DrivingPolicyHead(nn.Module):
    """Final control head that predicts steering, throttle, and brake."""

    def __init__(self, input_dim: int, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim + config.ego_state_dim),
            nn.Linear(input_dim + config.ego_state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(hidden_dim, config.action_dim)
        self.uncertainty_head = (
            nn.Linear(hidden_dim, config.action_dim) if config.predict_uncertainty else None
        )

    def forward(self, fused_state: Tensor, ego_state: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if ego_state is None:
            ego_state = fused_state.new_zeros(fused_state.shape[0], self.config.ego_state_dim)

        hidden = self.backbone(torch.cat((fused_state, ego_state), dim=-1))
        raw_actions = self.action_head(hidden)
        steering = torch.tanh(raw_actions[:, :1])
        throttle_brake = torch.sigmoid(raw_actions[:, 1:])
        actions = torch.cat((steering, throttle_brake), dim=-1)

        uncertainty = None
        if self.uncertainty_head is not None:
            uncertainty = torch.exp(self.uncertainty_head(hidden)).clamp(1e-4, 50.0)

        return actions, uncertainty


class HybridDrivingModel(nn.Module):
    """Research-grade hybrid architecture with explicit perception-policy fusion."""

    def __init__(
        self,
        config: HybridModelConfig | None = None,
        sam_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or HybridModelConfig()
        self.vision_encoder = DAVE2VisionEncoder(variant=self.config.vit_variant)
        self.perception = SAMPerceptionWrapper(
            config=self.config.perception,
            sam_backbone=sam_backbone,
        )
        control_dim = len(self.perception.control_extractor.FEATURE_NAMES)
        self.fusion = HybridFusionModule(
            config=self.config.fusion,
            vision_dim=self.vision_encoder.config.embed_dim,
            scene_dim=self.config.perception.embedding_dim,
            control_dim=control_dim,
        )
        self.policy_head = DrivingPolicyHead(
            input_dim=self.config.fusion.fusion_dim,
            config=self.config.policy,
        )

    def forward(
        self,
        images: Tensor,
        ego_state: Tensor | None = None,
        track_memory: TrackMemory | None = None,
        delta_t: float = 1.0,
    ) -> DrivingPolicyOutput:
        vision_cls, vision_tokens = self.vision_encoder.forward_features(images)
        perception_output = self.perception(images, track_memory=track_memory, delta_t=delta_t)

        fused_state = self.fusion(
            vision_tokens=torch.cat((vision_cls.unsqueeze(1), vision_tokens), dim=1),
            scene_tokens=perception_output.scene_tokens,
            control_features=perception_output.control.vector,
        )
        actions, uncertainty = self.policy_head(fused_state, ego_state=ego_state)
        return DrivingPolicyOutput(
            actions=actions,
            action_uncertainty=uncertainty,
            fused_state=fused_state,
            perception=perception_output,
        )


def build_hybrid_model(
    config: HybridModelConfig | None = None,
    sam_backbone: nn.Module | None = None,
) -> HybridDrivingModel:
    return HybridDrivingModel(config=config, sam_backbone=sam_backbone)
