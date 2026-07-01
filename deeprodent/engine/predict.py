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
Inference / prediction engine.

Wraps the DeepRodent model for single-image and video inference, and feeds
raw predictions into the post-processing aggregation engine (trajectory
tracking, occupancy heatmaps, behavioral-state tagging) described in the
paper's abstract and Section on downstream analytics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from deeprodent.models.deeprodent import DeepRodent
from deeprodent.utils.viz import build_trajectory, classify_behavior_state, occupancy_heatmap


class Predictor:
    """High-level inference wrapper around a trained DeepRodent model."""

    def __init__(self, model: DeepRodent, device: str = "cuda", img_size: int = 640, conf_thresh: float = 0.25):
        self.model = model.to(device).eval()
        self.device = device
        self.img_size = img_size
        self.conf_thresh = conf_thresh

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        if cv2 is not None:
            image = cv2.resize(image, (self.img_size, self.img_size))
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_image(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """Run a single forward pass and return raw multi-task outputs."""
        x = self._preprocess(image)
        return self.model(x)

    @torch.no_grad()
    def predict_video(
        self,
        frames: List[np.ndarray],
        centers_extractor: Optional[callable] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference over a sequence of frames and assemble downstream
        behavioral analytics (trajectory, occupancy heatmap, behavior states).

        Args:
            frames: list of HxWx3 RGB frames.
            centers_extractor: optional callable(outputs) -> (x, y) instance
                center in pixel coordinates. Defaults to a simple centroid
                derived from the highest-confidence detection channel.
        """
        centers, aspect_ratios = [], []

        for frame in frames:
            outputs = self.predict_image(frame)
            if centers_extractor is not None:
                cx, cy, ar = centers_extractor(outputs)
            else:
                cx, cy, ar = self._default_center_extractor(outputs)
            centers.append((cx, cy))
            aspect_ratios.append(ar)

        trajectory = build_trajectory(centers)
        heatmap = occupancy_heatmap(centers, (self.img_size, self.img_size))
        states = classify_behavior_state(trajectory, np.array(aspect_ratios))

        return {
            "trajectory": trajectory,
            "heatmap": heatmap,
            "behavior_states": np.array(states),
        }

    def _default_center_extractor(self, outputs: Dict[str, torch.Tensor]):
        """Fallback centroid extraction from the raw detection map (argmax objectness cell)."""
        det = outputs["det"][0]  # (A*(5+C), H, W)
        obj_channel = det[4]  # first anchor's objectness logit, index 4 (after 4 box coords)
        h, w = obj_channel.shape
        flat_idx = torch.argmax(obj_channel)
        gy, gx = divmod(int(flat_idx.item()), w)
        cx = (gx + 0.5) / w * self.img_size
        cy = (gy + 0.5) / h * self.img_size
        aspect_ratio = 1.0  # placeholder; replace with decoded box w/h once NMS is wired up
        return cx, cy, aspect_ratio
