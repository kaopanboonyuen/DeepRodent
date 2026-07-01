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
Oriented Bounding Box (OBB) geometry helpers implementing the parameterization
and rotation transform of Section 2.3:

    o_i = (x_i, y_i, w_i, h_i, theta_i)

    R(theta) = [[cos(theta), -sin(theta)],
                [sin(theta),  cos(theta)]]
"""

from __future__ import annotations

from typing import List

import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    """R(theta) as defined in Eq. (5)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def obb_to_polygon(obb: np.ndarray) -> np.ndarray:
    """
    Convert an oriented box (x, y, w, h, theta) to its 4 corner points.

    Returns an (4, 2) array of corner coordinates in the same units as
    (x, y, w, h).
    """
    x, y, w, h, theta = obb
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
        dtype=np.float32,
    )
    R = rotation_matrix(theta)
    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _sutherland_hodgman_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Clip `subject` polygon against convex `clip` polygon."""

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return p2
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return [px, py]

    output = list(subject)
    for i in range(len(clip)):
        if not output:
            break
        a, b = clip[i], clip[(i + 1) % len(clip)]
        input_list, output = output, []
        for j in range(len(input_list)):
            curr, prev = input_list[j], input_list[j - 1]
            curr_in, prev_in = inside(curr, a, b), inside(prev, a, b)
            if curr_in:
                if not prev_in:
                    output.append(intersect(prev, curr, a, b))
                output.append(curr)
            elif prev_in:
                output.append(intersect(prev, curr, a, b))
    return np.array(output, dtype=np.float32) if output else np.zeros((0, 2), dtype=np.float32)


def rotated_iou(obb_a: np.ndarray, obb_b: np.ndarray) -> float:
    """
    Exact rotated IoU via Sutherland-Hodgman polygon clipping:

        RIoU = A(o_p ∩ o_g) / A(o_p ∪ o_g)

    Used for evaluation (as opposed to the differentiable Gaussian-Wasserstein
    approximation used during training in `losses.RotatedIoULoss`).
    """
    poly_a = obb_to_polygon(obb_a)
    poly_b = obb_to_polygon(obb_b)

    inter_poly = _sutherland_hodgman_clip(poly_a, poly_b)
    inter_area = polygon_area(inter_poly) if len(inter_poly) >= 3 else 0.0

    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def rotated_iou_matrix(obbs_a: List[np.ndarray], obbs_b: List[np.ndarray]) -> np.ndarray:
    """Pairwise rotated IoU matrix between two sets of OBBs."""
    mat = np.zeros((len(obbs_a), len(obbs_b)), dtype=np.float32)
    for i, a in enumerate(obbs_a):
        for j, b in enumerate(obbs_b):
            mat[i, j] = rotated_iou(a, b)
    return mat
