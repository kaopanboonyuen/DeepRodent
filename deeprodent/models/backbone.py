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
Multi-scale feature integration backbone.

Implements the layer-wise feature extraction stack

    F_l = phi_l(F_{l-1}),  l = 1, ..., L

described in Section 2.2 of the paper, followed by the scale-aware
feature aggregation module of Section 2.5:

    F_agg = sum_s alpha_s * F_s,   alpha_s = softmax(gamma_s)

The backbone is intentionally lightweight (CSP-style blocks) so that the
reference implementation trains and runs at interactive speed on a single
GPU while remaining architecturally faithful to the multi-branch design
described in the paper. Swap in any YOLO-style backbone (v8/v9/v10/v11/v12)
by implementing the same `forward(x) -> List[Tensor]` interface.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def conv_bn_act(in_ch: int, out_ch: int, k: int = 3, s: int = 1, g: int = 1) -> nn.Sequential:
    """Standard Conv -> BatchNorm -> SiLU block."""
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, groups=g, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class CSPBlock(nn.Module):
    """A lightweight Cross-Stage-Partial residual block (phi_l in the paper)."""

    def __init__(self, in_ch: int, out_ch: int, n_bottlenecks: int = 2):
        super().__init__()
        hidden = out_ch // 2
        self.reduce = conv_bn_act(in_ch, hidden, k=1)
        self.blocks = nn.Sequential(
            *[nn.Sequential(conv_bn_act(hidden, hidden, k=3), conv_bn_act(hidden, hidden, k=3))
              for _ in range(n_bottlenecks)]
        )
        self.skip = conv_bn_act(in_ch, hidden, k=1)
        self.fuse = conv_bn_act(hidden * 2, out_ch, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_main = self.blocks(self.reduce(x))
        branch_skip = self.skip(x)
        return self.fuse(torch.cat([branch_main, branch_skip], dim=1))


class DownStage(nn.Module):
    """A downsampling stage that halves spatial resolution and increases channels."""

    def __init__(self, in_ch: int, out_ch: int, n_bottlenecks: int = 2):
        super().__init__()
        self.down = conv_bn_act(in_ch, out_ch, k=3, s=2)
        self.csp = CSPBlock(out_ch, out_ch, n_bottlenecks=n_bottlenecks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.csp(self.down(x))


class ScaleAwareFusion(nn.Module):
    """
    Scale-aware fusion module (Section 2.5):

        F_agg = sum_{s=1}^{S} alpha_s * F_s,   alpha_s = exp(gamma_s) / sum_k exp(gamma_k)

    Feature maps at different scales are first resized to a common
    resolution/channel-width before being combined with learned,
    softmax-normalized scalar weights `gamma`.
    """

    def __init__(self, channels: List[int], out_ch: int, target_stride: int = 16):
        super().__init__()
        self.target_stride = target_stride
        self.projections = nn.ModuleList([conv_bn_act(c, out_ch, k=1) for c in channels])
        self.gamma = nn.Parameter(torch.zeros(len(channels)))
        self.post = conv_bn_act(out_ch, out_ch, k=3)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # Use the middle-scale feature map as the common reference resolution.
        ref_h, ref_w = feats[len(feats) // 2].shape[-2:]

        alpha = torch.softmax(self.gamma, dim=0)
        agg = None
        for i, f in enumerate(feats):
            proj = self.projections[i](f)
            if proj.shape[-2:] != (ref_h, ref_w):
                proj = nn.functional.interpolate(proj, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            weighted = alpha[i] * proj
            agg = weighted if agg is None else agg + weighted
        return self.post(agg)


class MultiScaleBackbone(nn.Module):
    """
    Multi-scale feature integration backbone.

    Produces a pyramid of feature maps [P3, P4, P5] (strides 8/16/32) and a
    scale-aggregated feature map F_agg used by all task-specific heads.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth_multiples=(1, 2, 2, 1)):
        super().__init__()
        c1, c2, c3, c4 = [base_channels * m for m in (1, 2, 4, 8)]

        self.stem = conv_bn_act(in_channels, c1, k=3, s=2)          # stride 2
        self.stage1 = DownStage(c1, c2, n_bottlenecks=depth_multiples[0])   # stride 4
        self.stage2 = DownStage(c2, c3, n_bottlenecks=depth_multiples[1])   # stride 8  -> P3
        self.stage3 = DownStage(c3, c4, n_bottlenecks=depth_multiples[2])   # stride 16 -> P4
        self.stage4 = DownStage(c4, c4 * 2, n_bottlenecks=depth_multiples[3])  # stride 32 -> P5

        self.out_channels = [c3, c4, c4 * 2]
        self.fusion = ScaleAwareFusion(self.out_channels, out_ch=c4)
        self.fused_channels = c4

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        pyramid = [p3, p4, p5]
        f_agg = self.fusion(pyramid)
        return pyramid, f_agg
