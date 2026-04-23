"""SAM-style perception stack for hybrid autonomous driving research."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import PerceptionConfig


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    centers = boxes[..., :2]
    sizes = boxes[..., 2:].clamp_min(1e-4)
    half_sizes = sizes / 2.0
    top_left = (centers - half_sizes).clamp(0.0, 1.0)
    bottom_right = (centers + half_sizes).clamp(0.0, 1.0)
    return torch.cat((top_left, bottom_right), dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    top_left = boxes[..., :2]
    bottom_right = boxes[..., 2:]
    centers = (top_left + bottom_right) / 2.0
    sizes = (bottom_right - top_left).clamp_min(1e-4)
    return torch.cat((centers, sizes), dim=-1)


def pairwise_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0.0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clamp_min(0.0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0.0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clamp_min(0.0)
    union = area_a[:, None] + area_b[None, :] - intersection_area
    return intersection_area / union.clamp_min(1e-6)


def build_coordinate_grid(height: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
    ys = torch.linspace(0.0, 1.0, height, device=device)
    xs = torch.linspace(0.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_y, grid_x


@dataclass(slots=True)
class DetectionOutput:
    boxes: Tensor
    class_logits: Tensor
    objectness_logits: Tensor
    track_embeddings: Tensor

    @property
    def scores(self) -> Tensor:
        return torch.sigmoid(self.objectness_logits.squeeze(-1))


@dataclass(slots=True)
class SegmentationOutput:
    logits: Tensor

    @property
    def probabilities(self) -> Tensor:
        return torch.softmax(self.logits, dim=1)


@dataclass(slots=True)
class RoadOutput:
    logits: Tensor

    @property
    def probabilities(self) -> Tensor:
        return torch.sigmoid(self.logits)


@dataclass(slots=True)
class LaneOutput:
    lane_logits: Tensor
    boundary_logits: Tensor
    left_boundary_x: Tensor
    right_boundary_x: Tensor
    visibility: Tensor
    anchor_y: Tensor
    lane_embeddings: Tensor


@dataclass(slots=True)
class TrackMemory:
    boxes: Tensor
    embeddings: Tensor
    velocities: Tensor
    scores: Tensor
    ids: Tensor
    ages: Tensor
    misses: Tensor
    active_mask: Tensor


@dataclass(slots=True)
class TrackingOutput:
    track_ids: Tensor
    track_boxes: Tensor
    track_velocities: Tensor
    track_scores: Tensor
    memory: TrackMemory


@dataclass(slots=True)
class ControlFeatures:
    vector: Tensor
    names: tuple[str, ...]


@dataclass(slots=True)
class PerceptionOutput:
    scene_feature_map: Tensor
    scene_tokens: Tensor
    detections: DetectionOutput
    segmentation: SegmentationOutput
    road: RoadOutput
    lanes: LaneOutput
    tracking: TrackingOutput
    control: ControlFeatures


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


class SAMBackboneAdapter(nn.Module):
    """Adapter around a three-scale SAM-style encoder.

    When a real SAM backbone is unavailable, this falls back to a lightweight
    three-stage convolutional encoder that preserves the expected interface.
    """

    def __init__(
        self,
        config: PerceptionConfig,
        external_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.external_backbone = external_backbone

        stage_dims = config.backbone_dims
        self.stem = ConvNormAct(3, stage_dims[0], kernel_size=7, stride=2, padding=3)
        self.stage1 = nn.Sequential(
            ResidualBlock(stage_dims[0]),
            ResidualBlock(stage_dims[0]),
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(stage_dims[0], stage_dims[1], stride=2),
            ResidualBlock(stage_dims[1]),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(stage_dims[1], stage_dims[2], stride=2),
            ResidualBlock(stage_dims[2]),
            ResidualBlock(stage_dims[2]),
        )

    def _normalize_external_output(self, output: object) -> tuple[Tensor, Tensor, Tensor]:
        features: list[Tensor]
        if isinstance(output, dict):
            features = [feature for _, feature in sorted(output.items()) if torch.is_tensor(feature)]
        elif isinstance(output, (list, tuple)):
            features = [feature for feature in output if torch.is_tensor(feature)]
        elif torch.is_tensor(output):
            features = [output]
        else:
            raise TypeError("Unsupported SAM backbone output type.")

        if len(features) < 3:
            raise ValueError("Expected at least three multi-scale features from the SAM backbone.")

        return features[-3], features[-2], features[-1]

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.external_backbone is not None:
            return self._normalize_external_output(self.external_backbone(images))

        stage1 = self.stage1(self.stem(images))
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        return stage1, stage2, stage3


class FeaturePyramidAggregator(nn.Module):
    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__()
        fusion_dim = config.embedding_dim
        self.lateral_layers = nn.ModuleList(
            [
                nn.Conv2d(config.backbone_dims[0], fusion_dim, kernel_size=1),
                nn.Conv2d(config.backbone_dims[1], fusion_dim, kernel_size=1),
                nn.Conv2d(config.backbone_dims[2], fusion_dim, kernel_size=1),
            ]
        )
        self.refine = nn.Sequential(
            ConvNormAct(fusion_dim, fusion_dim),
            ResidualBlock(fusion_dim),
        )

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        stage1, stage2, stage3 = features
        p3 = self.lateral_layers[2](stage3)
        p2 = self.lateral_layers[1](stage2) + F.interpolate(
            p3,
            size=stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.lateral_layers[0](stage1) + F.interpolate(
            p2,
            size=stage1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(p1)


class SceneTokenizer(nn.Module):
    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__()
        self.token_grid = config.scene_token_grid
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, feature_map: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(feature_map, self.token_grid)
        tokens = pooled.flatten(start_dim=2).transpose(1, 2)
        return self.norm(self.projection(tokens))


class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(in_channels // 2, out_channels * 4)
        self.decoder = nn.Sequential(
            ConvNormAct(in_channels, hidden),
            ResidualBlock(hidden),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, feature_map: Tensor, output_size: tuple[int, int]) -> Tensor:
        logits = self.decoder(feature_map)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class QueryDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        normalized_queries = self.self_norm(queries)
        attended_queries, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + self.dropout(attended_queries)

        normalized_queries = self.cross_norm(queries)
        attended_memory, _ = self.cross_attention(
            normalized_queries,
            memory,
            memory,
            need_weights=False,
        )
        queries = queries + self.dropout(attended_memory)
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        return queries


class ObjectDetectionHead(nn.Module):
    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__()
        embed_dim = config.embedding_dim
        self.query_embeddings = nn.Embedding(config.num_detection_queries, embed_dim)
        self.decoder_layers = nn.ModuleList(
            [
                QueryDecoderLayer(embed_dim=embed_dim, num_heads=8, dropout=0.1),
                QueryDecoderLayer(embed_dim=embed_dim, num_heads=8, dropout=0.1),
            ]
        )
        self.class_head = nn.Linear(embed_dim, config.num_object_classes)
        self.objectness_head = nn.Linear(embed_dim, 1)
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        self.track_embedding_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, config.track_embedding_dim),
        )
        self.memory_projection = nn.Linear(embed_dim, embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)

    def forward(self, scene_tokens: Tensor) -> DetectionOutput:
        batch_size = scene_tokens.shape[0]
        queries = self.query_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        memory = self.memory_projection(scene_tokens)

        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        queries = self.query_norm(queries)
        boxes = box_cxcywh_to_xyxy(torch.sigmoid(self.box_head(queries)))
        class_logits = self.class_head(queries)
        objectness_logits = self.objectness_head(queries)
        track_embeddings = F.normalize(self.track_embedding_head(queries), dim=-1)
        return DetectionOutput(
            boxes=boxes,
            class_logits=class_logits,
            objectness_logits=objectness_logits,
            track_embeddings=track_embeddings,
        )


class LaneDetectionModule(nn.Module):
    """Lane representation with dense boundaries plus anchor-wise geometry."""

    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__()
        self.anchor_count = config.lane_anchor_count
        self.lane_decoder = SegmentationDecoder(config.embedding_dim, out_channels=1)
        self.boundary_decoder = SegmentationDecoder(config.embedding_dim, out_channels=2)
        self.embedding_projection = nn.Conv2d(
            config.embedding_dim,
            config.track_embedding_dim,
            kernel_size=1,
        )
        self.anchor_pool = nn.AdaptiveAvgPool2d((self.anchor_count, 1))
        self.anchor_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, 4),
        )

    def forward(self, feature_map: Tensor, output_size: tuple[int, int]) -> LaneOutput:
        lane_logits = self.lane_decoder(feature_map, output_size)
        boundary_logits = self.boundary_decoder(feature_map, output_size)
        lane_embeddings = self.embedding_projection(feature_map)

        anchor_features = self.anchor_pool(feature_map).squeeze(-1).transpose(1, 2)
        anchor_predictions = self.anchor_head(anchor_features)

        left_candidates = torch.sigmoid(anchor_predictions[..., 0])
        right_candidates = torch.sigmoid(anchor_predictions[..., 1])
        visibility = torch.sigmoid(anchor_predictions[..., 2:])

        left_boundary_x = torch.minimum(left_candidates, right_candidates)
        right_boundary_x = torch.maximum(left_candidates, right_candidates)
        anchor_y = torch.linspace(
            1.0,
            0.0,
            self.anchor_count,
            device=feature_map.device,
        )

        return LaneOutput(
            lane_logits=lane_logits,
            boundary_logits=boundary_logits,
            left_boundary_x=left_boundary_x,
            right_boundary_x=right_boundary_x,
            visibility=visibility,
            anchor_y=anchor_y,
            lane_embeddings=lane_embeddings,
        )


class TrackingModule(nn.Module):
    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__()
        self.max_tracks = config.max_tracks
        self.score_threshold = config.track_score_threshold
        self.match_threshold = config.track_match_threshold
        self.embedding_dim = config.track_embedding_dim
        self.register_buffer("next_track_id", torch.tensor(1, dtype=torch.long), persistent=False)

    def _allocate_memory(self, batch_size: int, device: torch.device) -> TrackMemory:
        shape = (batch_size, self.max_tracks)
        return TrackMemory(
            boxes=torch.zeros(batch_size, self.max_tracks, 4, device=device),
            embeddings=torch.zeros(batch_size, self.max_tracks, self.embedding_dim, device=device),
            velocities=torch.zeros(batch_size, self.max_tracks, 2, device=device),
            scores=torch.zeros(shape, device=device),
            ids=torch.full(shape, -1, dtype=torch.long, device=device),
            ages=torch.zeros(shape, device=device),
            misses=torch.zeros(shape, device=device),
            active_mask=torch.zeros(shape, dtype=torch.bool, device=device),
        )

    def _new_track_id(self, device: torch.device) -> Tensor:
        track_id = self.next_track_id.to(device=device).clone()
        self.next_track_id += 1
        return track_id

    def _greedy_match(self, similarity: Tensor) -> list[tuple[int, int]]:
        matches: list[tuple[int, int]] = []
        if similarity.numel() == 0:
            return matches

        matrix = similarity.clone()
        while matrix.numel() > 0:
            score, flat_index = matrix.view(-1).max(dim=0)
            if score < self.match_threshold:
                break

            det_index = int(flat_index // matrix.shape[1])
            track_index = int(flat_index % matrix.shape[1])
            matches.append((det_index, track_index))
            matrix[det_index, :] = -1.0
            matrix[:, track_index] = -1.0

        return matches

    def forward(
        self,
        detections: DetectionOutput,
        memory: TrackMemory | None = None,
        delta_t: float = 1.0,
    ) -> TrackingOutput:
        boxes = detections.boxes
        scores = detections.scores
        embeddings = detections.track_embeddings
        batch_size, num_queries, _ = boxes.shape
        device = boxes.device

        if memory is None:
            memory = self._allocate_memory(batch_size, device)

        new_memory = self._allocate_memory(batch_size, device)
        assigned_track_ids = torch.full((batch_size, num_queries), -1, dtype=torch.long, device=device)
        assigned_velocities = torch.zeros(batch_size, num_queries, 2, device=device)
        assigned_boxes = boxes.clone()
        assigned_scores = scores.clone()

        for batch_index in range(batch_size):
            detection_mask = scores[batch_index] >= self.score_threshold
            detection_indices = torch.nonzero(detection_mask, as_tuple=False).flatten()
            if detection_indices.numel() == 0:
                continue

            current_boxes = boxes[batch_index, detection_indices]
            current_embeddings = embeddings[batch_index, detection_indices]
            active_track_mask = memory.active_mask[batch_index]
            active_track_indices = torch.nonzero(active_track_mask, as_tuple=False).flatten()

            unmatched_detection_indices = detection_indices.tolist()
            matched_current_indices: set[int] = set()
            matched_track_slots: set[int] = set()

            if active_track_indices.numel() > 0:
                previous_boxes = memory.boxes[batch_index, active_track_indices]
                previous_embeddings = memory.embeddings[batch_index, active_track_indices]
                iou_matrix = pairwise_iou(current_boxes, previous_boxes)
                cosine_matrix = torch.matmul(current_embeddings, previous_embeddings.transpose(0, 1))
                similarity = 0.7 * iou_matrix + 0.3 * ((cosine_matrix + 1.0) / 2.0)

                for detection_local, track_local in self._greedy_match(similarity):
                    detection_global = int(detection_indices[detection_local].item())
                    track_slot = int(active_track_indices[track_local].item())
                    previous_box = memory.boxes[batch_index, track_slot]
                    current_box = boxes[batch_index, detection_global]
                    previous_center = box_xyxy_to_cxcywh(previous_box.unsqueeze(0))[0, :2]
                    current_center = box_xyxy_to_cxcywh(current_box.unsqueeze(0))[0, :2]
                    velocity = (current_center - previous_center) / max(delta_t, 1e-3)

                    new_memory.boxes[batch_index, track_slot] = current_box
                    new_memory.embeddings[batch_index, track_slot] = embeddings[batch_index, detection_global]
                    new_memory.velocities[batch_index, track_slot] = velocity
                    new_memory.scores[batch_index, track_slot] = scores[batch_index, detection_global]
                    new_memory.ids[batch_index, track_slot] = memory.ids[batch_index, track_slot]
                    new_memory.ages[batch_index, track_slot] = memory.ages[batch_index, track_slot] + 1
                    new_memory.misses[batch_index, track_slot] = 0
                    new_memory.active_mask[batch_index, track_slot] = True

                    assigned_track_ids[batch_index, detection_global] = memory.ids[batch_index, track_slot]
                    assigned_velocities[batch_index, detection_global] = velocity
                    matched_current_indices.add(detection_global)
                    matched_track_slots.add(track_slot)

            for track_slot in active_track_indices.tolist():
                if track_slot in matched_track_slots:
                    continue
                miss_count = memory.misses[batch_index, track_slot] + 1
                if miss_count > 2:
                    continue
                new_memory.boxes[batch_index, track_slot] = memory.boxes[batch_index, track_slot]
                new_memory.embeddings[batch_index, track_slot] = memory.embeddings[batch_index, track_slot]
                new_memory.velocities[batch_index, track_slot] = memory.velocities[batch_index, track_slot]
                new_memory.scores[batch_index, track_slot] = memory.scores[batch_index, track_slot] * 0.95
                new_memory.ids[batch_index, track_slot] = memory.ids[batch_index, track_slot]
                new_memory.ages[batch_index, track_slot] = memory.ages[batch_index, track_slot] + 1
                new_memory.misses[batch_index, track_slot] = miss_count
                new_memory.active_mask[batch_index, track_slot] = True

            free_slots = torch.nonzero(~new_memory.active_mask[batch_index], as_tuple=False).flatten().tolist()
            for detection_global in unmatched_detection_indices:
                if detection_global in matched_current_indices or not free_slots:
                    continue

                track_slot = free_slots.pop(0)
                new_track_id = self._new_track_id(device)
                new_memory.boxes[batch_index, track_slot] = boxes[batch_index, detection_global]
                new_memory.embeddings[batch_index, track_slot] = embeddings[batch_index, detection_global]
                new_memory.scores[batch_index, track_slot] = scores[batch_index, detection_global]
                new_memory.ids[batch_index, track_slot] = new_track_id
                new_memory.ages[batch_index, track_slot] = 1
                new_memory.active_mask[batch_index, track_slot] = True

                assigned_track_ids[batch_index, detection_global] = new_track_id

        return TrackingOutput(
            track_ids=assigned_track_ids,
            track_boxes=assigned_boxes,
            track_velocities=assigned_velocities,
            track_scores=assigned_scores,
            memory=new_memory,
        )


class ControlFeatureExtractor(nn.Module):
    """Converts perception outputs into policy-ready geometric and risk features."""

    FEATURE_NAMES = (
        "lane_center_offset",
        "heading_deviation",
        "lane_width",
        "lane_visibility",
        "lane_curvature",
        "drivable_area_ratio",
        "drivable_center_offset",
        "drivable_area_confidence",
        "object_proximity_risk",
        "nearest_object_lateral_offset",
        "ttc_proxy",
    )

    def forward(
        self,
        road: RoadOutput,
        lanes: LaneOutput,
        detections: DetectionOutput,
        tracking: TrackingOutput,
    ) -> ControlFeatures:
        lane_visibility = lanes.visibility.mean(dim=-1)
        lane_center = (lanes.left_boundary_x + lanes.right_boundary_x) / 2.0
        lane_width = (lanes.right_boundary_x - lanes.left_boundary_x).clamp_min(0.0)
        weighted_center = (lane_center * lane_visibility).sum(dim=1) / lane_visibility.sum(dim=1).clamp_min(1e-3)
        lane_center_offset = weighted_center - 0.5

        top_center = lane_center[:, -1]
        bottom_center = lane_center[:, 0]
        heading_deviation = torch.atan2(bottom_center - top_center, torch.ones_like(bottom_center))
        lane_curvature = (lane_center[:, :-2] - 2 * lane_center[:, 1:-1] + lane_center[:, 2:]).abs().mean(dim=1)
        mean_lane_width = (lane_width * lane_visibility).sum(dim=1) / lane_visibility.sum(dim=1).clamp_min(1e-3)
        visibility_score = lane_visibility.mean(dim=1)

        road_probabilities = road.probabilities.squeeze(1)
        batch_size, height, width = road_probabilities.shape
        grid_y, grid_x = build_coordinate_grid(height, width, road_probabilities.device)
        area_ratio = road_probabilities.mean(dim=(1, 2))
        x_mass = (road_probabilities * grid_x.unsqueeze(0)).sum(dim=(1, 2))
        total_mass = road_probabilities.sum(dim=(1, 2)).clamp_min(1e-6)
        drivable_center_offset = x_mass / total_mass - 0.5
        lower_half = road_probabilities[:, height // 2 :, :]
        drivable_confidence = lower_half.mean(dim=(1, 2))

        scores = detections.scores
        boxes = detections.boxes
        box_widths = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0)
        box_heights = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)
        areas = box_widths * box_heights
        centers_x = (boxes[..., 0] + boxes[..., 2]) / 2.0
        center_prior = 1.0 - (centers_x - 0.5).abs() * 2.0
        object_risk = scores * areas * center_prior.clamp_min(0.0)
        top_risk, top_indices = object_risk.max(dim=1)
        nearest_object_offset = centers_x.gather(1, top_indices.unsqueeze(-1)).squeeze(-1) - 0.5

        tracked_boxes = tracking.track_boxes
        tracked_velocities = tracking.track_velocities
        tracked_box_heights = (tracked_boxes[..., 3] - tracked_boxes[..., 1]).clamp_min(1e-4)
        approach_rate = tracked_velocities[..., 1].clamp_min(0.0)
        ttc_candidates = (1.0 / tracked_box_heights) / approach_rate.clamp_min(1e-3)
        valid_ttc = torch.where(tracking.track_ids >= 0, ttc_candidates, torch.full_like(ttc_candidates, 1e3))
        ttc_proxy = valid_ttc.min(dim=1).values.clamp(max=1e3)

        feature_vector = torch.stack(
            (
                lane_center_offset,
                heading_deviation,
                mean_lane_width,
                visibility_score,
                lane_curvature,
                area_ratio,
                drivable_center_offset,
                drivable_confidence,
                top_risk,
                nearest_object_offset,
                ttc_proxy,
            ),
            dim=-1,
        )

        return ControlFeatures(vector=feature_vector, names=self.FEATURE_NAMES)


class SAMPerceptionWrapper(nn.Module):
    """Full perception module that exposes dense and structured road-scene signals."""

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        sam_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PerceptionConfig()
        self.backbone = SAMBackboneAdapter(self.config, external_backbone=sam_backbone)
        self.fpn = FeaturePyramidAggregator(self.config)
        self.scene_tokenizer = SceneTokenizer(self.config)
        self.segmentation_head = SegmentationDecoder(
            self.config.embedding_dim,
            self.config.num_segmentation_classes,
        )
        self.road_head = SegmentationDecoder(self.config.embedding_dim, out_channels=1)
        self.lane_head = LaneDetectionModule(self.config)
        self.object_head = ObjectDetectionHead(self.config)
        self.tracker = TrackingModule(self.config)
        self.control_extractor = ControlFeatureExtractor()

    def forward(
        self,
        images: Tensor,
        track_memory: TrackMemory | None = None,
        delta_t: float = 1.0,
    ) -> PerceptionOutput:
        multi_scale_features = self.backbone(images)
        scene_feature_map = self.fpn(multi_scale_features)
        scene_tokens = self.scene_tokenizer(scene_feature_map)
        output_size = tuple(images.shape[-2:])

        detections = self.object_head(scene_tokens)
        segmentation = SegmentationOutput(logits=self.segmentation_head(scene_feature_map, output_size))
        road = RoadOutput(logits=self.road_head(scene_feature_map, output_size))
        lanes = self.lane_head(scene_feature_map, output_size)
        tracking = self.tracker(detections, memory=track_memory, delta_t=delta_t)
        control = self.control_extractor(road, lanes, detections, tracking)

        return PerceptionOutput(
            scene_feature_map=scene_feature_map,
            scene_tokens=scene_tokens,
            detections=detections,
            segmentation=segmentation,
            road=road,
            lanes=lanes,
            tracking=tracking,
            control=control,
        )
