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
Command-line entry point for running DeepRodent inference on an image or
video and exporting downstream behavioral analytics.

Usage:
    python scripts/predict.py --checkpoint checkpoints/deeprodent_epoch100.pt --source video.mp4 --out out/
    python scripts/predict.py --checkpoint checkpoints/deeprodent_epoch100.pt --source frame.jpg --out out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from deeprodent.engine.predict import Predictor
from deeprodent.models.deeprodent import DeepRodent, DeepRodentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepRodent inference.")
    parser.add_argument("--config", type=str, default="configs/deeprodent.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="Path to an image or video file.")
    parser.add_argument("--out", type=str, default="outputs", help="Directory to write analytics artifacts to.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_frames(source: str) -> list:
    path = Path(source)
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video/image I/O. `pip install opencv-python`.")

    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return [img]

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = DeepRodentConfig(**cfg["model"])
    model = DeepRodent(model_cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])

    predictor = Predictor(model, device=args.device, img_size=cfg["data"]["img_size"])
    frames = read_frames(args.source)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = predictor.predict_video(frames)
    np.save(out_dir / "trajectory.npy", results["trajectory"])
    np.save(out_dir / "heatmap.npy", results["heatmap"])
    np.save(out_dir / "behavior_states.npy", results["behavior_states"])

    if cv2 is not None:
        heatmap_vis = (results["heatmap"] * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(out_dir / "occupancy_heatmap.png"), heatmap_color)

    print(f"Processed {len(frames)} frame(s). Analytics saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
