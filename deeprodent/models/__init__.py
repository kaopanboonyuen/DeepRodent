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
"""Model sub-package: backbone, multi-task heads, and the DeepRodent wrapper."""

from deeprodent.models.backbone import MultiScaleBackbone  # noqa: F401
from deeprodent.models.heads import (  # noqa: F401
    DetectionHead,
    OBBHead,
    SegmentationHead,
    TemporalEmbeddingHead,
)
from deeprodent.models.deeprodent import DeepRodent  # noqa: F401
