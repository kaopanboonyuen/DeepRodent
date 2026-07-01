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
"""setup.py — packaging metadata for `pip install -e .`"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="deeprodent",
    version="1.0.0",
    description="DeepRodent: A Robust and Generalizable Vision Framework for Automated Rodent Monitoring in Experimental Biology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Teerapong Panboonyuen",
    author_email="teerapong.panboonyuen@gmail.com",
    url="https://github.com/kaopanboonyuen/DeepRodent",
    license="MIT",
    packages=find_packages(exclude=["tests", "scripts", "docs", "assets"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "PyYAML>=6.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "matplotlib>=3.7.0", "tqdm>=4.66.0"],
        "demo": ["gradio>=4.10.0", "huggingface_hub>=0.20.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
