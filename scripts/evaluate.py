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
Command-line entry point for evaluating a trained DeepRodent checkpoint.

Usage:
    python scripts/evaluate.py --config configs/deeprodent.yaml --checkpoint checkpoints/deeprodent_epoch100.pt
"""

from __future__ import annotations

import argparse
import json

import torch
import yaml
from torch.utils.data import DataLoader

from deeprodent.data.dataset import RodentSegDataset, collate_fn
from deeprodent.engine.evaluate import Evaluator
from deeprodent.models.deeprodent import DeepRodent, DeepRodentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepRodent model.")
    parser.add_argument("--config", type=str, default="configs/deeprodent.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = DeepRodentConfig(**cfg["model"])
    model = DeepRodent(model_cfg)

    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {state.get('epoch', '?')})")

    dataset = RodentSegDataset(
        root=cfg["data"]["root"],
        split=args.split,
        img_size=cfg["data"]["img_size"],
        num_classes=cfg["data"]["num_classes"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    evaluator = Evaluator(model, loader, device=args.device, conf_thresh=cfg["eval"]["conf_thresh"])
    report = evaluator.evaluate()

    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
