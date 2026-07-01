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
Data augmentation pipeline (Section 4.6):

    I' = T_geo(T_photo(I))

Implements:
    * Random horizontal flip
    * Random rotation (+-30 deg)
    * Scale jittering (0.5 ~ 1.5)
    * HSV color perturbation
    * Motion blur augmentation
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class DeepRodentAugmentor:
    """Composable geometric + photometric augmentation, T_geo(T_photo(I))."""

    def __init__(
        self,
        hflip_p: float = 0.5,
        rotate_deg: float = 30.0,
        scale_range=(0.5, 1.5),
        hsv_gains=(0.015, 0.7, 0.4),
        motion_blur_p: float = 0.2,
        enabled: bool = True,
    ):
        self.hflip_p = hflip_p
        self.rotate_deg = rotate_deg
        self.scale_range = scale_range
        self.hsv_gains = hsv_gains
        self.motion_blur_p = motion_blur_p
        self.enabled = enabled and cv2 is not None

    # ---------------------------------------------------------------- #
    # Photometric transforms: T_photo
    # ---------------------------------------------------------------- #
    def _hsv_perturb(self, image: np.ndarray) -> np.ndarray:
        h_gain, s_gain, v_gain = self.hsv_gains
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        gains = [
            1 + random.uniform(-h_gain, h_gain),
            1 + random.uniform(-s_gain, s_gain),
            1 + random.uniform(-v_gain, v_gain),
        ]
        hsv[..., 0] = np.clip(hsv[..., 0] * gains[0], 0, 179)
        hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _motion_blur(self, image: np.ndarray, ksize: int = 7) -> np.ndarray:
        kernel = np.zeros((ksize, ksize))
        kernel[(ksize - 1) // 2, :] = np.ones(ksize)
        kernel /= ksize
        return cv2.filter2D(image, -1, kernel)

    def apply_photometric(self, image: np.ndarray) -> np.ndarray:
        image = self._hsv_perturb(image)
        if random.random() < self.motion_blur_p:
            image = self._motion_blur(image)
        return image

    # ---------------------------------------------------------------- #
    # Geometric transforms: T_geo
    # ---------------------------------------------------------------- #
    def apply_geometric(self, sample: Dict) -> Dict:
        image = sample["image"]
        h, w = image.shape[:2]

        # Random horizontal flip
        if random.random() < self.hflip_p:
            image = np.ascontiguousarray(image[:, ::-1, :])
            if sample["boxes"].shape[0] > 0:
                sample["boxes"][:, 0] = 1.0 - sample["boxes"][:, 0]
            if sample["obb"].shape[0] > 0:
                sample["obb"][:, 0] = 1.0 - sample["obb"][:, 0]
                sample["obb"][:, 4] = -sample["obb"][:, 4]
            if sample["masks"].shape[0] > 0:
                sample["masks"] = sample["masks"][:, :, ::-1]

        # Random rotation +- rotate_deg and scale jitter, applied jointly via affine warp.
        angle = random.uniform(-self.rotate_deg, self.rotate_deg)
        scale = random.uniform(*self.scale_range)
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT101)

        if sample["masks"].shape[0] > 0:
            warped_masks = np.stack(
                [cv2.warpAffine(m, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT) for m in sample["masks"]],
                axis=0,
            )
            sample["masks"] = warped_masks
            # Update OBB angle (theta) to reflect the applied rotation.
            if sample["obb"].shape[0] > 0:
                sample["obb"][:, 4] -= np.deg2rad(angle)
            # Boxes are re-derived downstream from masks if needed; keep as-is for a light-weight pipeline.

        sample["image"] = image
        return sample

    def __call__(self, sample: Dict) -> Dict:
        if not self.enabled:
            return sample
        sample["image"] = self.apply_photometric(sample["image"])
        sample = self.apply_geometric(sample)
        return sample
