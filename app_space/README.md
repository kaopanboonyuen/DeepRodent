---
title: DeepRodent Demo
emoji: 🐭
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: "5.38.2"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: 'Automated Rodent Monitoring in Experimental Biology'
---
<div align="center">

# 🐭 DeepRodent

### A Robust and Generalizable Vision Framework for Automated Rodent Monitoring in Experimental Biology

**Author: Teerapong Panboonyuen**

Department of Computer Science, College of Computing, Khon Kaen University, Thailand · PBYAIL (PBY Artificial Intelligence Laboratory), Bangkok · Faculty of Engineering, Chulalongkorn University, Thailand

📧 [teerapong.panboonyuen@gmail.com](mailto:teerapong.panboonyuen@gmail.com)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://kaopanboonyuen.github.io/DeepRodent)
[![Code](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/kaopanboonyuen/DeepRodent)
[![Weights](https://img.shields.io/badge/🤗%20Hugging%20Face-Weights-yellow)](https://huggingface.co/kaopanboonyuen/DeepRodent)
[![Demo](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-orange)](https://huggingface.co/spaces/kaopanboonyuen/DeepRodent-Demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

</div>

---

## Overview

Continuous, non-invasive behavioral monitoring of rodents in laboratory environments is fundamental to experimental biology, neuroscience, and pharmacological phenotyping. Existing automated vision methods often fail to generalize across diverse laboratory settings due to varying illumination, perspective distortions, cage geometries, and high-density animal interactions that cause severe occlusions.

**DeepRodent** is a robust, unified deep learning framework engineered specifically for precise and generalizable multi-task rodent monitoring. Moving beyond conventional axis-aligned bounding boxes, DeepRodent incorporates a multi-path prediction architecture that simultaneously optimizes:

- 📦 **Object Detection** — axis-aligned bounding boxes
- 🔄 **Oriented Bounding Boxes (OBB)** — rotation-aware localization for curled / rotated rodents
- 🎭 **Pixel-level Instance Segmentation** — precise silhouette masks under occlusion

...paired with a **post-processing aggregation engine** that converts raw spatial coordinates into actionable downstream biological analytics: trajectory tracking, behavioral-state classification (grooming, rearing, locomotion), and spatial occupancy heatmaps.

<p align="center">
  <img src="assets/architecture_placeholder.svg" alt="DeepRodent architecture overview" width="720"/>
</p>

---

## Key Results

| Method | Backbone | Precision | Recall | mAP₅₀ | mAP₅₀₋₉₅ | FPS |
|---|---|---|---|---|---|---|
| YOLOv8-Seg | Nano | 91.7 | 89.6 | 92.8 | 73.9 | **188** |
| YOLO11-Seg | Small | 93.5 | 92.1 | 94.2 | 77.4 | 161 |
| YOLO12-Seg | Small | 93.8 | 92.5 | 94.4 | 78.2 | 156 |
| **DeepRodent (Ours)** | YOLO Family | **95.4** | **94.1** | **96.2** | **84.6** | 154 |

DeepRodent is **detector-agnostic**: plugging it into YOLOv8–YOLO12 backbones yields a consistent **+2.6 to +3.1 mAP** improvement across the board, while maintaining real-time inference speed (154 FPS) suitable for continuous behavioral monitoring. Full benchmark tables (cross-environment generalization, ablations, SOTA comparison) are reported in the paper.

---

## Repository Structure

```
DeepRodent/
├── app.py                      # Gradio demo (mirrors the HF Spaces demo)
├── setup.py                    # pip install -e .
├── requirements.txt
├── configs/
│   └── deeprodent.yaml         # all hyperparameters from the paper
├── deeprodent/                 # main library
│   ├── models/
│   │   ├── backbone.py         # multi-scale feature integration backbone
│   │   ├── heads.py            # detection / OBB / segmentation / temporal heads
│   │   └── deeprodent.py       # unified DeepRodent model wrapper
│   ├── losses/
│   │   └── losses.py           # full multi-task loss (Eq. 11–22 in the paper)
│   ├── data/
│   │   ├── dataset.py          # polygon-annotation dataset loader
│   │   └── augment.py          # geometric + photometric augmentation
│   ├── engine/
│   │   ├── train.py            # Trainer (SGD + momentum + cosine LR)
│   │   ├── evaluate.py         # Evaluator (mAP50 / mAP50-95 / FPS)
│   │   └── predict.py          # Predictor (image / video inference)
│   └── utils/
│       ├── metrics.py          # precision / recall / mAP utilities
│       ├── obb_utils.py        # rotated-box geometry, exact rotated IoU
│       ├── viz.py              # trajectory / heatmap / behavior-state utils
│       └── seed.py             # reproducibility seeding
├── scripts/
│   ├── train.py                # CLI training entry point
│   ├── evaluate.py             # CLI evaluation entry point
│   ├── predict.py              # CLI inference entry point
│   ├── export_weights.py       # export a clean release checkpoint
│   └── make_toy_dataset.py     # synthetic dataset for pipeline smoke-testing
├── tests/
│   ├── test_model.py
│   └── test_losses.py
├── docs/
│   └── ARCHITECTURE.md         # equation-by-equation mapping to code
├── CITATION.cff
├── LICENSE
└── README.md
```

Every module carries a `DeepRodent` / `Author: Teerapong Panboonyuen` header for clear provenance and reviewer traceability.

---

## Installation

```bash
git clone https://github.com/kaopanboonyuen/DeepRodent.git
cd DeepRodent

python -m venv .venv && source .venv/bin/activate   # optional but recommended

pip install -e .
# or, for demo (Gradio) support:
pip install -e ".[demo]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.1 (CUDA recommended for training). See [`requirements.txt`](requirements.txt) for the full list.

---

## Quickstart

### 1. Get a dataset

DeepRodent expects the standard YOLO-style polygon segmentation layout:

```
data/DeepRodentDataset/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt      # "c x1 y1 x2 y2 ... xn yn" per line, normalized [0,1]
```

Don't have data on hand yet? Generate a small synthetic dataset to smoke-test the full pipeline:

```bash
python scripts/make_toy_dataset.py --root ./data/DeepRodentDataset --n-per-split 30
```

> **Note:** the benchmark numbers reported in the paper were obtained on a **private**, multi-setting laboratory dataset collected with the Faculty of Pharmaceutical Sciences, Khon Kaen University. The toy generator above is provided purely for verifying that the code runs end-to-end — it is not a substitute for the paper's dataset.

### 2. Train

```bash
python scripts/train.py --config configs/deeprodent.yaml --epochs 100 --seed 42
```

Key hyperparameters (from Section 4 of the paper) are already set in [`configs/deeprodent.yaml`](configs/deeprodent.yaml):

```
eta_0 = 1e-3, momentum = 0.937, weight_decay = 5e-4
cosine annealing LR schedule
mixed-precision (FP16) training
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
  --config configs/deeprodent.yaml \
  --checkpoint checkpoints/deeprodent_epoch100.pt \
  --split test
```

Reports Precision, Recall, mAP₅₀, mAP₅₀₋₉₅, and FPS.

### 4. Run inference

```bash
python scripts/predict.py \
  --checkpoint checkpoints/deeprodent_epoch100.pt \
  --source path/to/video.mp4 \
  --out outputs/
```

Produces trajectory arrays, an occupancy heatmap, and per-frame behavioral-state tags in `outputs/`.

### 5. Launch the interactive demo locally

```bash
pip install -e ".[demo]"
python app.py
```

This is the same interface powering the [Hugging Face Spaces demo](https://huggingface.co/spaces/kaopanboonyuen/DeepRodent-Demo).

### Minimal Python API

```python
import torch
from deeprodent import DeepRodent
from deeprodent.models.deeprodent import DeepRodentConfig

model = DeepRodent(DeepRodentConfig())
model.eval()

x = torch.randn(1, 3, 640, 640)  # a batch of RGB frames
with torch.no_grad():
    outputs = model(x)

outputs["det"]        # axis-aligned detection logits   -> B_t
outputs["obb"]        # oriented box params (x,y,w,h,θ) -> O_t
outputs["seg_proto"]  # segmentation prototypes         -> M_t
outputs["embed"]      # behavioral embedding             -> E_t
```

---

## Method Summary

DeepRodent's overall prediction function follows:

```
F_theta(I_t) = { B_t, M_t, O_t, E_t }
```

where a shared **multi-scale feature integration backbone** (CSP-style blocks + scale-aware softmax fusion) feeds four task-specific heads — detection, OBB, instance segmentation, and a temporal behavioral embedding — trained jointly under a unified objective:

```
L_DeepRodent = λ₁L_cls + λ₂L_box + λ₃L_seg + λ₄L_obb + λ₅L_temp
             + β·L_KL + λ₆L_domain + λ₇L_temp
```

combining focal segmentation loss, IoU box loss, a Gaussian-Wasserstein rotated-IoU loss for OBB regression, KL-divergence regularization, uncertainty-guided reweighting, and a cross-domain robustness (feature-moment matching) term for generalization across laboratory settings.

For an equation-by-equation mapping from the paper to the codebase, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Reproducibility

To ensure full reproducibility and support open science, we release:

| Resource | Link |
|---|---|
| 🔗 Project page | https://kaopanboonyuen.github.io/DeepRodent |
| 💻 Source code | https://github.com/kaopanboonyuen/DeepRodent |
| 🤗 Pretrained weights | https://huggingface.co/kaopanboonyuen/DeepRodent |
| 🚀 Interactive demo | https://huggingface.co/spaces/kaopanboonyuen/DeepRodent-Demo |

All ablation results in the paper are reported across **3 random seeds** with the multi-seed averaging protocol implemented in `Evaluator.multi_seed_summary`. Set `PYTHONHASHSEED`, NumPy, and PyTorch seeds via `deeprodent.utils.seed.set_seed(...)` before training to reproduce a given run.

---

## Ethical Considerations

DeepRodent is intended solely as an **assistive research framework** and is not designed to replace expert veterinary oversight or certified behavioral assessment by trained experimental biologists. The underlying study used a private, non-invasive laboratory video dataset (secondary analysis of recorded observation clips only); no housing conditions were altered and no invasive procedures were performed for the purpose of data collection. All animal care and handling from the primary data source were conducted under approved IACUC protocols, in accordance with the ARRIVE guidelines and the 3Rs principles (Replacement, Reduction, Refinement).

DeepRodent should be treated as a **decision-support** tool requiring expert oversight, continual monitoring, and multi-center validation prior to broader deployment in experimental biology workflows.

---

## Contributing

Issues and pull requests are welcome! Please open an issue on [GitHub](https://github.com/kaopanboonyuen/DeepRodent/issues) to discuss significant changes before submitting a PR. Run the test suite before opening a PR:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Citation

If you find DeepRodent useful for your research, please cite:

```bibtex
@article{panboonyuen2026deeprodent,
  title   = {DeepRodent: A Robust and Generalizable Vision Framework for Automated Rodent Monitoring in Experimental Biology},
  author  = {Panboonyuen, Teerapong},
  year    = {2026},
  url     = {https://github.com/kaopanboonyuen/DeepRodent}
}
```