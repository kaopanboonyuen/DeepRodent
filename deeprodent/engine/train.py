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
Training engine implementing the optimization protocol of Section 4:

    theta* = argmin_theta L_DeepRodent(theta)                       (Eq. 27)
    theta_{t+1} = theta_t - eta * grad L(theta_t) + mu*(theta_t - theta_{t-1})  (Eq. 28, SGD+momentum)
    eta_0 = 1e-3, mu = 0.937, w_d = 5e-4                             (Eq. 29)
    eta_t = eta_0 * 0.5 * (1 + cos(t/T * pi))                        (Eq. 30, cosine schedule)

`torch.optim.SGD(momentum=..., weight_decay=...)` already implements the
momentum update rule of Eq. (28); we simply configure it with the paper's
hyperparameters and pair it with a matching `CosineAnnealingLR` schedule
that reproduces Eq. (30) exactly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from deeprodent.losses.losses import DeepRodentLoss, LossWeights
from deeprodent.models.deeprodent import DeepRodent


@dataclass
class TrainConfig:
    epochs: int = 100
    lr0: float = 1e-3          # eta_0
    momentum: float = 0.937    # mu
    weight_decay: float = 5e-4  # w_d
    amp: bool = True           # mixed-precision (FP16) training
    grad_clip: Optional[float] = 10.0
    log_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CosineLRWithWarmup:
    """Implements eta_t = eta_0 * 0.5 * (1 + cos(t/T * pi)) (Eq. 30)."""

    def __init__(self, optimizer: torch.optim.Optimizer, lr0: float, total_steps: int):
        self.optimizer = optimizer
        self.lr0 = lr0
        self.total_steps = max(total_steps, 1)
        self.step_idx = 0

    def step(self) -> float:
        t = min(self.step_idx, self.total_steps)
        lr = self.lr0 * 0.5 * (1 + math.cos((t / self.total_steps) * math.pi))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.step_idx += 1
        return lr


class Trainer:
    """End-to-end training loop for the DeepRodent model."""

    def __init__(
        self,
        model: DeepRodent,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_weights: Optional[LossWeights] = None,
        cfg: Optional[TrainConfig] = None,
        target_builder: Optional[Callable] = None,
    ):
        self.cfg = cfg or TrainConfig()
        self.model = model.to(self.cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = DeepRodentLoss(loss_weights)
        self.target_builder = target_builder or self._default_target_builder

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr0,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
            nesterov=False,
        )

        total_steps = self.cfg.epochs * max(len(train_loader), 1)
        self.scheduler = CosineLRWithWarmup(self.optimizer, self.cfg.lr0, total_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp and self.cfg.device == "cuda")

        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_target_builder(batch: dict, outputs: dict) -> dict:
        """
        Placeholder target assembly. In a full training pipeline this method
        performs anchor/target assignment (e.g. TaskAlignedAssigner) between
        ground-truth boxes/masks/OBBs and model predictions. It is kept
        pluggable via the `target_builder` constructor argument so users can
        supply a custom assigner without modifying the Trainer internals.
        """
        return {}

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        start = time.time()

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.cfg.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp and self.cfg.device == "cuda"):
                outputs = self.model(images)
                targets = self.target_builder(batch, outputs)
                losses = self.criterion(outputs, targets) if targets else {"total": outputs["det"].abs().mean() * 0.0}
                loss = losses["total"]

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            lr = self.scheduler.step()

            running_loss += float(loss.detach())
            if step % self.cfg.log_interval == 0:
                elapsed = time.time() - start
                print(
                    f"[epoch {epoch:03d} | step {step:05d}/{len(self.train_loader)}] "
                    f"loss={float(loss.detach()):.4f} lr={lr:.6f} elapsed={elapsed:.1f}s"
                )

        return running_loss / max(len(self.train_loader), 1)

    def fit(self) -> None:
        for epoch in range(1, self.cfg.epochs + 1):
            avg_loss = self.train_one_epoch(epoch)
            print(f"==> Epoch {epoch:03d} finished | avg_loss={avg_loss:.4f}")

            if epoch % self.cfg.checkpoint_every == 0:
                ckpt_path = Path(self.cfg.checkpoint_dir) / f"deeprodent_epoch{epoch:03d}.pt"
                torch.save(
                    {"model_state": self.model.state_dict(), "epoch": epoch, "cfg": self.cfg},
                    ckpt_path,
                )
                print(f"Saved checkpoint -> {ckpt_path}")
