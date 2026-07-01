# =============================================================================
#   ____                  ____           _            _
#  |  _ \  ___  ___ _ __ |  _ \ ___   __| | ___ _ __ | |_
#  | | | |/ _ \/ _ \ '_ \| |_) / _ \ / _` |/ _ \ '_ \| __|
#  | |_| |  __/  __/ |_) |  _ < (_) | (_| |  __/ | | | |_
#  |____/ \___|\___| .__/|_| \_\___/ \__,_|\___|_| |_|\__|
#                   |_|
#
#  DeepRodent: A Robust and Generalizable Vision Framework for Automated
#              Rodent Monitoring in Experimental Biology
# -----------------------------------------------------------------------------
#  Author       : Teerapong Panboonyuen
#  Contact      : teerapong.panboonyuen@gmail.com
#  Source Code  : https://github.com/kaopanboonyuen/DeepRodent
#  License      : MIT (see LICENSE)
# =============================================================================
"""
DeepRodent: unified multi-task model.

Implements the overall prediction function of Section 2.1 (Eq. 1):

    F_theta(I_t) = { B_t, M_t, O_t, E_t }

by combining a `MultiScaleBackbone` with detector-agnostic task heads for
axis-aligned detection, oriented bounding boxes, instance segmentation, and
temporal behavioral embeddings.

The model is "detector-agnostic" in the sense described in the paper: the
backbone can be swapped for any YOLO-style feature extractor as long as it
returns a feature pyramid + a scale-aggregated feature map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from deeprodent.models.backbone import MultiScaleBackbone
from deeprodent.models.heads import (
    DetectionHead,
    MotionAwareUpdate,
    OBBHead,
    SegmentationHead,
    TemporalEmbeddingHead,
)


@dataclass
class DeepRodentConfig:
    """Configuration for the DeepRodent model. Mirrors configs/deeprodent.yaml."""

    in_channels: int = 3
    base_channels: int = 32
    num_classes: int = 1
    num_anchors: int = 1
    num_prototypes: int = 32
    embed_dim: int = 64
    motion_dim: int = 4


class DeepRodent(nn.Module):
    """
    Unified single-stage, multi-head network for rodent monitoring.

    Forward pass returns a dictionary with keys:
        - "det":        (B, A*(5+C), H, W)   raw detection logits (B_t branch)
        - "obb":        (B, A*5, H, W)       oriented box params  (O_t branch)
        - "seg_proto":  (B, K, 2H, 2W)       segmentation prototypes (M_t branch)
        - "embed":      (B, embed_dim)       behavioral embedding    (E_t branch)
        - "feat_agg":   (B, C, H, W)         scale-aggregated backbone feature
    """

    def __init__(self, cfg: Optional[DeepRodentConfig] = None):
        super().__init__()
        self.cfg = cfg or DeepRodentConfig()

        self.backbone = MultiScaleBackbone(
            in_channels=self.cfg.in_channels,
            base_channels=self.cfg.base_channels,
        )
        fused_ch = self.backbone.fused_channels

        self.det_head = DetectionHead(
            fused_ch, num_classes=self.cfg.num_classes, num_anchors=self.cfg.num_anchors
        )
        self.obb_head = OBBHead(fused_ch, num_anchors=self.cfg.num_anchors)
        self.seg_head = SegmentationHead(fused_ch, num_prototypes=self.cfg.num_prototypes)
        self.temporal_head = TemporalEmbeddingHead(fused_ch, embed_dim=self.cfg.embed_dim)
        self.motion_update = MotionAwareUpdate(
            embed_dim=self.cfg.embed_dim, motion_dim=self.cfg.motion_dim
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, f_agg = self.backbone(x)

        det = self.det_head(f_agg)
        obb = self.obb_head(f_agg)
        seg_proto = self.seg_head(f_agg)
        embed = self.temporal_head(f_agg)

        return {
            "det": det,
            "obb": obb,
            "seg_proto": seg_proto,
            "embed": embed,
            "feat_agg": f_agg,
        }

    @torch.no_grad()
    def track_step(
        self,
        x: torch.Tensor,
        prev_embed: torch.Tensor,
        motion_delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience inference step that also applies the motion-aware
        embedding update (Eq. 10) for online, frame-by-frame tracking.
        """
        outputs = self.forward(x)
        pooled_feat = nn.functional.adaptive_avg_pool2d(outputs["feat_agg"], 1).flatten(1)
        updated_embed = self.motion_update(prev_embed, pooled_feat[:, : prev_embed.shape[-1]], motion_delta)
        outputs["embed"] = updated_embed
        return outputs

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
