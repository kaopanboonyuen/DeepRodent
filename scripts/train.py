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
Command-line entry point for training DeepRodent.

Usage:
    python scripts/train.py --config configs/deeprodent.yaml
    python scripts/train.py --config configs/deeprodent.yaml --epochs 50 --seed 42
"""

from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from deeprodent.data.augment import DeepRodentAugmentor
from deeprodent.data.dataset import RodentSegDataset, collate_fn
from deeprodent.engine.train import Trainer, TrainConfig
from deeprodent.losses.losses import LossWeights
from deeprodent.models.deeprodent import DeepRodent, DeepRodentConfig
from deeprodent.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DeepRodent model.")
    parser.add_argument("--config", type=str, default="configs/deeprodent.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    model_cfg = DeepRodentConfig(**cfg["model"])
    model = DeepRodent(model_cfg)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state"])
        print(f"Resumed weights from {args.resume} (epoch {state.get('epoch', '?')})")

    print(f"DeepRodent initialized with {model.num_parameters():,} trainable parameters.")

    augmentor = DeepRodentAugmentor(**cfg["augment"])
    train_ds = RodentSegDataset(
        root=cfg["data"]["root"],
        split="train",
        img_size=cfg["data"]["img_size"],
        num_classes=cfg["data"]["num_classes"],
        transforms=augmentor,
    )
    val_ds = RodentSegDataset(
        root=cfg["data"]["root"],
        split="val",
        img_size=cfg["data"]["img_size"],
        num_classes=cfg["data"]["num_classes"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    train_cfg = TrainConfig(**cfg["train"])
    loss_weights = LossWeights(**cfg["loss"])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_weights=loss_weights,
        cfg=train_cfg,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
