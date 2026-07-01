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
"""Loss sub-package implementing the full DeepRodent objective (Section 2.7-2.12)."""

from deeprodent.losses.losses import (  # noqa: F401
    ClassificationLoss,
    CrossDomainRobustnessLoss,
    DeepRodentLoss,
    FocalSegmentationLoss,
    IoUBoxLoss,
    KLRegularizer,
    RotatedIoULoss,
    TemporalConsistencyLoss,
    UncertaintyReweighting,
)
