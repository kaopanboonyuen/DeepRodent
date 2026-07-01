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
"""Unit tests for each term of the DeepRodent multi-task loss."""

import torch

from deeprodent.losses.losses import (
    ClassificationLoss,
    DeepRodentLoss,
    FocalSegmentationLoss,
    IoUBoxLoss,
    KLRegularizer,
    RotatedIoULoss,
    TemporalConsistencyLoss,
)


def test_classification_loss_zero_when_perfect():
    loss_fn = ClassificationLoss()
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    loss = loss_fn(logits, targets)
    assert loss.item() < 1e-2


def test_iou_box_loss_perfect_overlap():
    loss_fn = IoUBoxLoss()
    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    loss = loss_fn(box, box.clone())
    assert loss.item() < 1e-5


def test_rotated_iou_loss_perfect_overlap():
    loss_fn = RotatedIoULoss()
    obb = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]])
    loss = loss_fn(obb, obb.clone())
    assert loss.item() < 1e-3


def test_focal_segmentation_loss_runs():
    loss_fn = FocalSegmentationLoss()
    pred = torch.sigmoid(torch.randn(2, 1, 16, 16))
    gt = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = loss_fn(pred, gt)
    assert loss.item() >= 0.0


def test_temporal_consistency_zero_when_static():
    loss_fn = TemporalConsistencyLoss()
    embeds = torch.ones(5, 2, 16)  # (T, B, D), constant across time
    loss = loss_fn(embeds)
    assert loss.item() < 1e-6


def test_kl_regularizer_zero_when_identical():
    loss_fn = KLRegularizer()
    p = torch.softmax(torch.randn(4, 3), dim=-1)
    loss = loss_fn(p, p.clone())
    assert loss.item() < 1e-4


def test_deeprodent_loss_aggregation():
    criterion = DeepRodentLoss()
    preds = {
        "cls_logits": torch.randn(2, 3),
        "boxes": torch.rand(2, 4),
        "obb": torch.rand(2, 5),
        "masks": torch.sigmoid(torch.randn(2, 1, 8, 8)),
        "embeds": torch.randn(3, 2, 16),
    }
    targets = {
        "cls": torch.softmax(torch.randn(2, 3), dim=-1),
        "boxes": torch.rand(2, 4),
        "obb": torch.rand(2, 5),
        "masks": (torch.rand(2, 1, 8, 8) > 0.5).float(),
    }
    losses = criterion(preds, targets)
    assert "total" in losses
    assert torch.isfinite(losses["total"])
