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
Task-specific prediction heads.

Implements the multi-branch decomposition of Section 2.2:

    F = F^cls (+) F^box (+) F^seg (+) F^obb

    * DetectionHead          -> objectness + class logits + axis-aligned box (B_t)
    * OBBHead                -> oriented box (x, y, w, h, theta)               (O_t)
    * SegmentationHead       -> per-instance prototype masks                   (M_t)
    * TemporalEmbeddingHead  -> behavioral embedding for temporal consistency  (E_t)

All heads consume the scale-aggregated feature map F_agg produced by
`MultiScaleBackbone`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_act(in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> nn.Sequential:
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class DetectionHead(nn.Module):
    """Predicts objectness, class scores, and axis-aligned box offsets (B_t)."""

    def __init__(self, in_ch: int, num_classes: int = 1, num_anchors: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.stem = conv_bn_act(in_ch, in_ch, k=3)
        # 4 box regressions (cx, cy, w, h) + 1 objectness + num_classes
        out_ch = num_anchors * (4 + 1 + num_classes)
        self.pred = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.stem(x))


class OBBHead(nn.Module):
    """
    Oriented bounding box head implementing:

        o_i = (x_i, y_i, w_i, h_i, theta_i)

    theta is regressed in [-pi/2, pi/2) via a tanh-scaled activation to
    resolve the periodic ambiguity discussed in Section 2.3.
    """

    def __init__(self, in_ch: int, num_anchors: int = 1):
        super().__init__()
        self.num_anchors = num_anchors
        self.stem = conv_bn_act(in_ch, in_ch, k=3)
        # (x, y, w, h, theta) per anchor
        self.pred = nn.Conv2d(in_ch, num_anchors * 5, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pred(self.stem(x))
        b, c, h, w = out.shape
        out = out.view(b, self.num_anchors, 5, h, w)
        xywh, theta = out[:, :, :4], out[:, :, 4:5]
        theta = torch.tanh(theta) * (torch.pi / 2)
        out = torch.cat([xywh, theta], dim=2).view(b, c, h, w)
        return out


class SegmentationHead(nn.Module):
    """
    Pixel-level instance segmentation head:

        M_i = sigmoid(W_m * F_seg(i))

    Predicts a bank of `num_prototypes` mask prototypes that are linearly
    combined (per-instance) with detection-head mask coefficients to yield
    the final instance masks — following the YOLACT-style prototype
    decomposition used by modern YOLO-segmentation heads.
    """

    def __init__(self, in_ch: int, num_prototypes: int = 32):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.stem = nn.Sequential(
            conv_bn_act(in_ch, in_ch, k=3),
            conv_bn_act(in_ch, in_ch, k=3),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proto = nn.Conv2d(in_ch, num_prototypes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.upsample(x)
        proto = self.proto(x)
        return torch.sigmoid(proto)

    @staticmethod
    def assemble_instance_masks(prototypes: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Combine prototypes (B, K, H, W) with per-instance coefficients
        (B, N, K) to produce instance masks (B, N, H, W), matching:

            M_i = sigmoid(W_m * F_seg(i))
        """
        b, k, h, w = prototypes.shape
        proto_flat = prototypes.view(b, k, h * w)
        masks = torch.bmm(coeffs, proto_flat)  # (B, N, H*W)
        masks = masks.view(b, -1, h, w)
        return torch.sigmoid(masks)


class TemporalEmbeddingHead(nn.Module):
    """
    Learned behavioral embedding head producing E_t, later regularized by
    the temporal-smoothness loss L_temp (Section 2.6):

        L_temp = sum_t || E_t - E_{t+1} ||_2^2
    """

    def __init__(self, in_ch: int, embed_dim: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, embed_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return self.fc(pooled)


class MotionAwareUpdate(nn.Module):
    """
    Motion-aware embedding update (Eq. 10):

        E_t = psi(E_{t-1}, F_t, delta_x_t)

    A small GRU-style gated update that fuses the previous embedding, the
    current spatial context, and an (optional) motion delta.
    """

    def __init__(self, embed_dim: int = 64, motion_dim: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + motion_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(embed_dim * 2 + motion_dim, embed_dim),
            nn.Tanh(),
        )

    def forward(self, prev_embed: torch.Tensor, curr_feat: torch.Tensor, motion_delta: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([prev_embed, curr_feat, motion_delta], dim=-1)
        z = self.gate(combined)
        cand = self.update(combined)
        return (1 - z) * prev_embed + z * cand
