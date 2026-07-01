#!/usr/bin/env python
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
Generates a small synthetic dataset in the DeepRodent polygon-annotation
format (Section 3) so the full pipeline — dataset loading, augmentation,
training loop, evaluation, and inference — can be smoke-tested end-to-end
without access to the private laboratory dataset used in the paper.

Usage:
    python scripts/make_toy_dataset.py --root ./data/DeepRodentDataset --n-per-split 20
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("OpenCV is required: pip install opencv-python") from e


def make_blob_polygon(cx: float, cy: float, r: float, n_pts: int = 16, jitter: float = 0.15) -> np.ndarray:
    """Generate an irregular blob polygon (normalized coords) approximating a rodent silhouette."""
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii = r * (1 + np.random.uniform(-jitter, jitter, size=n_pts))
    xs = cx + radii * np.cos(angles) * 1.4  # elongate horizontally, rodent-like
    ys = cy + radii * np.sin(angles) * 0.8
    return np.clip(np.stack([xs, ys], axis=1), 0.0, 1.0)


def render_sample(img_size: int, n_instances: int):
    image = np.full((img_size, img_size, 3), fill_value=random.randint(180, 230), dtype=np.uint8)
    # bedding texture noise
    noise = np.random.randint(-15, 15, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    polygons = []
    for _ in range(n_instances):
        cx, cy = np.random.uniform(0.15, 0.85, size=2)
        r = np.random.uniform(0.05, 0.12)
        poly = make_blob_polygon(cx, cy, r)
        polygons.append(poly)

        pts = (poly * img_size).astype(np.int32).reshape(-1, 1, 2)
        color = tuple(int(c) for c in np.random.randint(40, 90, size=3))
        cv2.fillPoly(image, [pts], color=color)

    return image, polygons


def write_split(root: Path, split: str, n_samples: int, img_size: int):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        n_instances = random.randint(1, 3)
        image, polygons = render_sample(img_size, n_instances)

        img_path = img_dir / f"{split}_{i:05d}.jpg"
        lbl_path = lbl_dir / f"{split}_{i:05d}.txt"

        cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        with open(lbl_path, "w") as f:
            for poly in polygons:
                coords = " ".join(f"{v:.6f}" for v in poly.flatten())
                f.write(f"0 {coords}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic DeepRodent-format toy dataset.")
    parser.add_argument("--root", type=str, default="./data/DeepRodentDataset")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--n-per-split", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root)
    for split, ratio in [("train", 1.0), ("val", 0.3), ("test", 0.3)]:
        n = int(args.n_per_split * ratio) if split != "train" else args.n_per_split
        write_split(root, split, max(n, 1), args.img_size)
        print(f"Wrote {max(n, 1)} synthetic samples to {root / 'images' / split}")

    print("Toy dataset generation complete. This is for pipeline smoke-testing only —")
    print("it does NOT reproduce the paper's private laboratory rodent benchmark.")


if __name__ == "__main__":
    main()
