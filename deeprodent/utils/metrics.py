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
Evaluation metrics: mAP@50, mAP@50-95, precision, recall (Section 4.8).

    mAP_50, mAP_50:95, Precision, Recall, FPS
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Pixel-wise IoU between two binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def precision_recall(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float = 0.5) -> Tuple[float, float]:
    """
    Simple greedy matching precision/recall at a single IoU threshold.

    pred_boxes / gt_boxes: (N, 4) arrays of (x1, y1, x2, y2).
    """
    if len(pred_boxes) == 0:
        return (0.0, 0.0 if len(gt_boxes) == 0 else 0.0)
    if len(gt_boxes) == 0:
        return (0.0, 1.0)

    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = _box_iou_xyxy(pb, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thresh and best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return precision, recall


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mean_average_precision(
    all_precisions: List[float],
    all_recalls: List[float],
) -> float:
    """
    11-point interpolated average precision given a list of (precision, recall)
    samples collected across confidence thresholds for a single class.
    """
    if not all_precisions:
        return 0.0
    recalls = np.array(all_recalls)
    precisions = np.array(all_precisions)
    order = np.argsort(recalls)
    recalls, precisions = recalls[order], precisions[order]

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


def map_50_95(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> float:
    """Average precision across IoU thresholds 0.50:0.05:0.95, matching mAP_50-95."""
    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for thr in thresholds:
        p, r = precision_recall(pred_boxes, gt_boxes, iou_thresh=thr)
        aps.append(mean_average_precision([p], [r]))
    return float(np.mean(aps)) if aps else 0.0
