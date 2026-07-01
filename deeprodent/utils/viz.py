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
Post-processing aggregation engine.

Translates raw per-frame spatial predictions into actionable downstream
biological analytics, as described in the abstract and introduction:

    * Trajectory tracking            -> `build_trajectory`
    * Spatial occupancy heatmaps     -> `occupancy_heatmap`
    * Behavioral state classification -> `classify_behavior_state` (heuristic
      placeholder based on displacement / bounding-box aspect ratio; swap in
      a trained classifier head for production use)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def build_trajectory(centers: List[Tuple[float, float]]) -> np.ndarray:
    """Stack a sequence of (x, y) instance centers into a trajectory array (T, 2)."""
    return np.array(centers, dtype=np.float32)


def occupancy_heatmap(centers: List[Tuple[float, float]], img_size: Tuple[int, int], sigma: float = 8.0) -> np.ndarray:
    """
    Build a spatial occupancy heatmap by splatting Gaussian kernels at each
    observed instance center over the course of a recording.
    """
    h, w = img_size
    heatmap = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]

    for cx, cy in centers:
        heatmap += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def classify_behavior_state(
    trajectory: np.ndarray,
    aspect_ratios: np.ndarray,
    speed_thresh: float = 2.0,
    elongation_thresh: float = 1.8,
) -> List[str]:
    """
    Lightweight heuristic behavioral-state tagger used to demonstrate the
    downstream analytics pipeline described in the abstract
    ("grooming, rearing, locomotion"). Replace with a trained temporal
    classifier head (e.g. built on top of `TemporalEmbeddingHead`) for
    research-grade behavioral phenotyping.
    """
    if len(trajectory) < 2:
        return ["stationary"] * len(trajectory)

    speeds = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    speeds = np.concatenate([[0.0], speeds])

    states = []
    for speed, ar in zip(speeds, aspect_ratios):
        if speed > speed_thresh:
            states.append("locomotion")
        elif ar > elongation_thresh:
            states.append("rearing")
        else:
            states.append("grooming")
    return states
