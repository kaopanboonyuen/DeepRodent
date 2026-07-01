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
#  Affiliations : Department of Computer Science, College of Computing,
#                 Khon Kaen University, Thailand
#                 PBYAIL (PBY Artificial Intelligence Laboratory), Bangkok
#                 Faculty of Engineering, Chulalongkorn University, Thailand
#  Contact      : teerapong.panboonyuen@gmail.com
#  Project Page : https://kaopanboonyuen.github.io/DeepRodent
#  Source Code  : https://github.com/kaopanboonyuen/DeepRodent
#  Weights      : https://huggingface.co/kaopanboonyuen/DeepRodent
#  Demo         : https://huggingface.co/spaces/kaopanboonyuen/DeepRodent-Demo
#  License      : MIT (see LICENSE)
# =============================================================================
"""
DeepRodent
==========

A unified multi-task vision framework for automated rodent monitoring,
supporting simultaneous:

    * Standard (axis-aligned) object detection
    * Oriented Bounding Box (OBB) regression
    * Pixel-level instance segmentation
    * Temporally-consistent behavioral embeddings

This package provides a reference PyTorch implementation of the model,
losses, dataset utilities, and training / evaluation engines described in
the DeepRodent paper.
"""

__title__ = "DeepRodent"
__author__ = "Teerapong Panboonyuen"
__email__ = "teerapong.panboonyuen@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"
__url__ = "https://github.com/kaopanboonyuen/DeepRodent"

from deeprodent.models.deeprodent import DeepRodent  # noqa: F401
