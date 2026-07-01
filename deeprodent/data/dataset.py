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
RodentSegDataset: loader for the DeepRodent polygon-annotation format
described in Section 3 (Dataset) of the paper.

Each image `xxx.jpg` is paired with a label file `xxx.txt` where every
line encodes one polygon instance as:

    c x1 y1 x2 y2 ... xn yn

with class id `c` (rodent = 0) and normalized vertex coordinates
(x_k, y_k) in [0, 1], matching Eq. (24) of the paper:

    Y = {c, x_1, y_1, x_2, y_2, ..., x_n, y_n}

This mirrors the standard Ultralytics YOLO-segmentation label format, so
datasets exported from common annotation tools (CVAT, Roboflow, Label
Studio) can be used directly without conversion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def polygon_to_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterize a normalized polygon (N, 2) into a binary mask of shape (H, W)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon.shape[0] < 3:
        return mask
    pts = polygon.copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts = pts.astype(np.int32).reshape(-1, 1, 2)
    if cv2 is not None:
        cv2.fillPoly(mask, [pts], color=1)
    else:  # pragma: no cover - fallback path if OpenCV is unavailable
        from PIL import Image, ImageDraw

        img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in pts.reshape(-1, 2)], outline=1, fill=1)
        mask = np.array(img, dtype=np.uint8)
    return mask


def polygon_to_obb(polygon: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Convert a polygon (N, 2, normalized) into an oriented box (x, y, w, h, theta) via minAreaRect."""
    if cv2 is None or polygon.shape[0] < 3:  # pragma: no cover
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)
        return (
            float((x_min + x_max) / 2),
            float((y_min + y_max) / 2),
            float(x_max - x_min),
            float(y_max - y_min),
            0.0,
        )
    pts = (polygon * 1000).astype(np.float32)  # scale for numerical stability
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    return cx / 1000, cy / 1000, w / 1000, h / 1000, np.deg2rad(angle)


class RodentSegDataset(Dataset):
    """
    Loads image + polygon-annotation pairs and produces detection boxes,
    OBB parameters, and rasterized instance masks for a given image size.

    Expected directory layout:

        root/
          images/{train,val,test}/*.jpg
          labels/{train,val,test}/*.txt
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        num_classes: int = 1,
        transforms: Optional[object] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.transforms = transforms

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.image_paths = sorted(
            [p for p in self.image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        ) if self.image_dir.exists() else []

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_labels(self, label_path: Path) -> List[Dict]:
        instances = []
        if not label_path.exists():
            return instances
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cls_id = int(float(parts[0]))
                coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                instances.append({"class_id": cls_id, "polygon": coords})
        return instances

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        if cv2 is not None:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:  # pragma: no cover
            from PIL import Image as PILImage

            image = np.array(PILImage.open(img_path).convert("RGB"))

        h, w = image.shape[:2]
        instances = self._load_labels(label_path)

        boxes, obbs, masks, classes = [], [], [], []
        for inst in instances:
            poly = inst["polygon"]
            x_min, y_min = poly.min(axis=0)
            x_max, y_max = poly.max(axis=0)
            cx, cy, bw, bh = (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min
            boxes.append([cx, cy, bw, bh])

            obbs.append(list(polygon_to_obb(poly)))
            masks.append(polygon_to_mask(poly, self.img_size, self.img_size))
            classes.append(inst["class_id"])

        sample = {
            "image": image,
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "obb": np.array(obbs, dtype=np.float32) if obbs else np.zeros((0, 5), dtype=np.float32),
            "masks": np.array(masks, dtype=np.uint8) if masks else np.zeros((0, self.img_size, self.img_size), dtype=np.uint8),
            "classes": np.array(classes, dtype=np.int64) if classes else np.zeros((0,), dtype=np.int64),
            "image_path": str(img_path),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return self._to_tensors(sample)

    def _to_tensors(self, sample: Dict) -> Dict[str, torch.Tensor]:
        image = sample["image"]
        if cv2 is not None:
            image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        return {
            "image": image,
            "boxes": torch.from_numpy(sample["boxes"]),
            "obb": torch.from_numpy(sample["obb"]),
            "masks": torch.from_numpy(sample["masks"]).float(),
            "classes": torch.from_numpy(sample["classes"]),
            "image_path": sample["image_path"],
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Custom collate function supporting a variable number of instances per image."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    return {
        "image": images,
        "boxes": [b["boxes"] for b in batch],
        "obb": [b["obb"] for b in batch],
        "masks": [b["masks"] for b in batch],
        "classes": [b["classes"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }
