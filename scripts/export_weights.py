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
Export a training checkpoint into a clean, release-ready weights file
(strips optimizer/scheduler state) suitable for uploading to Hugging Face
Hub or attaching to a GitHub release.

Usage:
    python scripts/export_weights.py --checkpoint checkpoints/deeprodent_epoch100.pt --out release/deeprodent.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deeprodent import __version__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a clean DeepRodent release checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="release/deeprodent.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")

    release_state = {
        "model_state": state["model_state"],
        "epoch": state.get("epoch"),
        "version": __version__,
        "framework": "DeepRodent",
        "author": "Teerapong Panboonyuen",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release_state, out_path)
    print(f"Exported release checkpoint -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
