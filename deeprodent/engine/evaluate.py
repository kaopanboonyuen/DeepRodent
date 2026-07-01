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
Evaluation engine (Section 4.8-4.9):

    mAP_50, mAP_50:95, Precision, Recall, FPS

Also supports the multi-seed ablation protocol:

    E[mAP] = (1/3) * sum_{i=1}^{3} mAP_i
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from deeprodent.models.deeprodent import DeepRodent
from deeprodent.utils.metrics import map_50_95, mean_average_precision, precision_recall


@dataclass
class EvalReport:
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    fps: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "mAP50": self.map50,
            "mAP50-95": self.map50_95,
            "fps": self.fps,
        }


class Evaluator:
    """Runs inference over a validation/test split and computes standard metrics."""

    def __init__(self, model: DeepRodent, data_loader: DataLoader, device: str = "cuda", conf_thresh: float = 0.25):
        self.model = model.to(device).eval()
        self.data_loader = data_loader
        self.device = device
        self.conf_thresh = conf_thresh

    @torch.no_grad()
    def evaluate(self, decode_fn=None) -> EvalReport:
        """
        Args:
            decode_fn: optional callable(outputs, conf_thresh) -> (pred_boxes, gt_boxes)
                       arrays of shape (N, 4) in xyxy format, used to compute
                       precision/recall/mAP. If omitted, only FPS is measured
                       (useful for quickly benchmarking architectural changes
                       before wiring up a full post-processing / NMS pipeline).
        """
        precisions, recalls, maps = [], [], []
        total_images = 0
        start = time.time()

        for batch in self.data_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            outputs = self.model(images)
            total_images += images.shape[0]

            if decode_fn is not None:
                pred_boxes, gt_boxes = decode_fn(outputs, batch, self.conf_thresh)
                p, r = precision_recall(pred_boxes, gt_boxes, iou_thresh=0.5)
                precisions.append(p)
                recalls.append(r)
                maps.append(map_50_95(pred_boxes, gt_boxes))

        elapsed = time.time() - start
        fps = total_images / elapsed if elapsed > 0 else 0.0

        report = EvalReport(
            precision=float(np.mean(precisions)) if precisions else 0.0,
            recall=float(np.mean(recalls)) if recalls else 0.0,
            map50=float(mean_average_precision(precisions, recalls)) if precisions else 0.0,
            map50_95=float(np.mean(maps)) if maps else 0.0,
            fps=fps,
        )
        return report

    @staticmethod
    def multi_seed_summary(map_scores: List[float]) -> Dict[str, float]:
        """E[mAP] = (1/3) * sum_i mAP_i, generalized to N seeds (Eq. 26)."""
        arr = np.array(map_scores, dtype=np.float64)
        return {"mean_map": float(arr.mean()), "std_map": float(arr.std()), "n_seeds": len(arr)}
