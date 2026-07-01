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
Custom multi-task loss function.

Implements every term of the DeepRodent objective described in the paper:

    L_total  = l1*L_cls + l2*L_box + l3*L_seg + l4*L_obb + l5*L_temp     (Eq. 11)
    L_cls    = -sum_i y_i log(p_i)                                       (Eq. 12)
    L_seg    = -sum_i (1-p_i)^gamma log(p_i)                             (Eq. 13, focal)
    L_box    = 1 - IoU(B_p, B_g)                                         (Eq. 14)
    L_obb    = 1 - RIoU(o_p, o_g)                                        (Eq. 15-16)
    L_KL     = sum_i P(i) log(P(i)/Q(i))                                 (Eq. 17)
    L_final  = L_total + beta * L_KL                                     (Eq. 18)
    u_i      = -sum_c p_ic log p_ic  (uncertainty)                       (Eq. 19)
    w_i      = 1 / (1 + exp(u_i))    (adaptive weighting)                (Eq. 20)
    L_domain = ||mu_s - mu_t||_2^2 + ||sigma_s - sigma_t||_2^2           (Eq. 21)
    L_DeepRodent = L_final + l6*L_domain + l7*L_temp                     (Eq. 22)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Individual loss terms
# --------------------------------------------------------------------------- #


class ClassificationLoss(nn.Module):
    """Cross-entropy classification loss (Eq. 12): L_cls = -sum_i y_i log(p_i)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()


class FocalSegmentationLoss(nn.Module):
    """Focal-enhanced segmentation loss (Eq. 13): L_seg = -sum_i (1-p_i)^gamma log(p_i)."""

    def __init__(self, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        pred_mask = pred_mask.clamp(self.eps, 1 - self.eps)
        pt = torch.where(gt_mask > 0.5, pred_mask, 1 - pred_mask)
        loss = -((1 - pt) ** self.gamma) * torch.log(pt)
        return loss.mean()


class IoUBoxLoss(nn.Module):
    """Axis-aligned IoU box loss (Eq. 14): L_box = 1 - |Bp ∩ Bg| / |Bp ∪ Bg|."""

    @staticmethod
    def box_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        # boxes as (cx, cy, w, h) -> convert to (x1, y1, x2, y2)
        def to_xyxy(b):
            cx, cy, w, h = b.unbind(-1)
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        p, t = to_xyxy(pred), to_xyxy(target)
        inter_x1 = torch.max(p[..., 0], t[..., 0])
        inter_y1 = torch.max(p[..., 1], t[..., 1])
        inter_x2 = torch.min(p[..., 2], t[..., 2])
        inter_y2 = torch.min(p[..., 3], t[..., 3])
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        area_p = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
        area_t = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
        union = area_p + area_t - inter_area + eps
        return inter_area / union

    def forward(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        iou = self.box_iou(pred_boxes, gt_boxes)
        return (1 - iou).mean()


class RotatedIoULoss(nn.Module):
    """
    Rotated IoU loss for OBB (Eq. 15-16):

        L_obb = 1 - RIoU(o_p, o_g),   RIoU = A(o_p ∩ o_g) / A(o_p ∪ o_g)

    Uses a differentiable Gaussian-Wasserstein approximation to rotated IoU
    (common practice for OBB regression) rather than explicit polygon
    clipping, trading a small amount of exactness for full differentiability
    and GPU efficiency.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _covariance(w: torch.Tensor, h: torch.Tensor, theta: torch.Tensor):
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        a = (w ** 2) / 12 * cos_t ** 2 + (h ** 2) / 12 * sin_t ** 2
        b = (w ** 2) / 12 * sin_t ** 2 + (h ** 2) / 12 * cos_t ** 2
        c = ((w ** 2) - (h ** 2)) / 12 * cos_t * sin_t
        return a, b, c

    def forward(self, pred_obb: torch.Tensor, gt_obb: torch.Tensor) -> torch.Tensor:
        # obb: (..., 5) = (x, y, w, h, theta)
        px, py, pw, ph, pt = pred_obb.unbind(-1)
        gx, gy, gw, gh, gt_ = gt_obb.unbind(-1)

        a1, b1, c1 = self._covariance(pw, ph, pt)
        a2, b2, c2 = self._covariance(gw, gh, gt_)

        # Gaussian Wasserstein distance between the two OBB "distributions".
        center_term = (px - gx) ** 2 + (py - gy) ** 2
        shape_term = (
            (a1 - a2) ** 2 + (b1 - b2) ** 2 + 2 * (c1 - c2) ** 2
        )
        wasserstein = center_term + shape_term
        riou_approx = 1.0 / (1.0 + torch.sqrt(wasserstein.clamp(min=self.eps)))
        return (1 - riou_approx).mean()


class KLRegularizer(nn.Module):
    """KL-divergence regularizer (Eq. 17): L_KL = sum_i P(i) log(P(i)/Q(i))."""

    def forward(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        return (p * torch.log(p / q)).sum(dim=-1).mean()


class TemporalConsistencyLoss(nn.Module):
    """Temporal smoothness loss (Eq. 9): L_temp = sum_t || E_t - E_{t+1} ||_2^2."""

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        # embeds: (T, B, D)
        diffs = embeds[1:] - embeds[:-1]
        return diffs.pow(2).sum(dim=-1).mean()


class UncertaintyReweighting(nn.Module):
    """
    Uncertainty-guided reweighting (Eq. 19-20):

        u_i = -sum_c p_ic log p_ic
        w_i = 1 / (1 + exp(u_i))
    """

    def forward(self, probs: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        probs = probs.clamp(min=eps)
        entropy = -(probs * torch.log(probs)).sum(dim=-1)
        weights = 1.0 / (1.0 + torch.exp(entropy))
        return weights


class CrossDomainRobustnessLoss(nn.Module):
    """Domain-alignment loss (Eq. 21): L_domain = ||mu_s-mu_t||^2 + ||sigma_s-sigma_t||^2."""

    def forward(self, feat_source: torch.Tensor, feat_target: torch.Tensor) -> torch.Tensor:
        mu_s, sigma_s = feat_source.mean(dim=0), feat_source.std(dim=0)
        mu_t, sigma_t = feat_target.mean(dim=0), feat_target.std(dim=0)
        return (mu_s - mu_t).pow(2).sum() + (sigma_s - sigma_t).pow(2).sum()


# --------------------------------------------------------------------------- #
# Combined objective
# --------------------------------------------------------------------------- #


@dataclass
class LossWeights:
    """Task weights lambda_1..lambda_7 and the KL weight beta (Eq. 11, 18, 22)."""

    cls: float = 1.0     # lambda_1
    box: float = 1.0     # lambda_2
    seg: float = 1.0     # lambda_3
    obb: float = 1.0     # lambda_4
    temp: float = 0.5    # lambda_5
    domain: float = 0.5  # lambda_6
    temp_final: float = 0.5  # lambda_7
    kl_beta: float = 0.1  # beta
    focal_gamma: float = 2.0


class DeepRodentLoss(nn.Module):
    """
    Full DeepRodent training objective (Eq. 22):

        L_DeepRodent = L_final + lambda_6 * L_domain + lambda_7 * L_temp
        L_final      = L_total + beta * L_KL
        L_total      = l1*L_cls + l2*L_box + l3*L_seg + l4*L_obb + l5*L_temp
    """

    def __init__(self, weights: Optional[LossWeights] = None):
        super().__init__()
        self.w = weights or LossWeights()

        self.cls_loss = ClassificationLoss()
        self.seg_loss = FocalSegmentationLoss(gamma=self.w.focal_gamma)
        self.box_loss = IoUBoxLoss()
        self.obb_loss = RotatedIoULoss()
        self.kl_loss = KLRegularizer()
        self.temp_loss = TemporalConsistencyLoss()
        self.domain_loss = CrossDomainRobustnessLoss()
        self.uncertainty = UncertaintyReweighting()

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds: dict with keys among
                {"cls_logits", "boxes", "obb", "masks", "embeds",
                 "cls_probs", "feat_source", "feat_target"}
            targets: matching dict of ground-truth tensors with keys among
                {"cls", "boxes", "obb", "masks", "kl_reference"}

        Returns:
            dict of individual loss terms plus the aggregated "total" loss.
        """
        losses: Dict[str, torch.Tensor] = {}
        device = next(iter(preds.values())).device
        total = torch.zeros((), device=device)

        if "cls_logits" in preds and "cls" in targets:
            l_cls = self.cls_loss(preds["cls_logits"], targets["cls"])
            losses["cls"] = l_cls
            total = total + self.w.cls * l_cls

        if "boxes" in preds and "boxes" in targets:
            l_box = self.box_loss(preds["boxes"], targets["boxes"])
            losses["box"] = l_box
            total = total + self.w.box * l_box

        if "masks" in preds and "masks" in targets:
            l_seg = self.seg_loss(preds["masks"], targets["masks"])
            losses["seg"] = l_seg
            total = total + self.w.seg * l_seg

        if "obb" in preds and "obb" in targets:
            l_obb = self.obb_loss(preds["obb"], targets["obb"])
            losses["obb"] = l_obb
            total = total + self.w.obb * l_obb

        if "embeds" in preds and preds["embeds"].dim() == 3:
            l_temp = self.temp_loss(preds["embeds"])
            losses["temp"] = l_temp
            total = total + self.w.temp * l_temp
        else:
            l_temp = torch.zeros((), device=device)
            losses["temp"] = l_temp

        l_final = total
        if "cls_probs" in preds and "kl_reference" in targets:
            l_kl = self.kl_loss(preds["cls_probs"], targets["kl_reference"])
            losses["kl"] = l_kl
            l_final = l_final + self.w.kl_beta * l_kl

        l_domain = torch.zeros((), device=device)
        if "feat_source" in preds and "feat_target" in preds:
            l_domain = self.domain_loss(preds["feat_source"], preds["feat_target"])
            losses["domain"] = l_domain

        l_deeprodent = l_final + self.w.domain * l_domain + self.w.temp_final * l_temp
        losses["total"] = l_deeprodent
        return losses
