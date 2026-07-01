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
"""Sanity tests for the DeepRodent model's forward pass and output shapes."""

import torch

from deeprodent.models.deeprodent import DeepRodent, DeepRodentConfig


def test_forward_pass_shapes():
    cfg = DeepRodentConfig(base_channels=8, num_classes=1, num_anchors=1, num_prototypes=8, embed_dim=16)
    model = DeepRodent(cfg)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert "det" in out and "obb" in out and "seg_proto" in out and "embed" in out
    assert out["embed"].shape == (2, 16)
    assert out["det"].dim() == 4
    assert out["obb"].dim() == 4
    assert out["seg_proto"].dim() == 4


def test_num_parameters_positive():
    model = DeepRodent(DeepRodentConfig(base_channels=8))
    assert model.num_parameters() > 0


def test_backbone_pyramid_strides():
    from deeprodent.models.backbone import MultiScaleBackbone

    backbone = MultiScaleBackbone(base_channels=8)
    x = torch.randn(1, 3, 256, 256)
    pyramid, f_agg = backbone(x)

    # Strides should be 8, 16, 32 relative to a 256-input.
    expected_sizes = [256 // 8, 256 // 16, 256 // 32]
    for feat, expected in zip(pyramid, expected_sizes):
        assert feat.shape[-1] == expected

    assert f_agg.shape[1] == backbone.fused_channels
