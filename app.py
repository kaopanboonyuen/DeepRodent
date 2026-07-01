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
Interactive Gradio demo for DeepRodent.

Powers the Hugging Face Spaces demo referenced in the paper:
    https://huggingface.co/spaces/kaopanboonyuen/DeepRodent-Demo

Run locally with:
    python app.py
"""

from __future__ import annotations

import numpy as np

try:
    import gradio as gr
except ImportError as e:  # pragma: no cover
    raise ImportError("Install the demo extras first: `pip install -e .[demo]`") from e

import torch

from deeprodent.engine.predict import Predictor
from deeprodent.models.deeprodent import DeepRodent, DeepRodentConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: for the public demo, replace this random-init model with a checkpoint
# downloaded from the Hugging Face Hub, e.g.:
#
#   from huggingface_hub import hf_hub_download
#   ckpt_path = hf_hub_download("kaopanboonyuen/DeepRodent", "deeprodent.pt")
#   state = torch.load(ckpt_path, map_location="cpu")
#   model.load_state_dict(state["model_state"])
_model = DeepRodent(DeepRodentConfig())
_predictor = Predictor(_model, device=DEVICE, img_size=640)


def run_inference(image: np.ndarray):
    if image is None:
        return None, "Please upload a lab-cage frame to run DeepRodent."

    outputs = _predictor.predict_image(image)
    seg_proto = outputs["seg_proto"][0].detach().cpu().numpy()
    proto_preview = (seg_proto[0] * 255).astype(np.uint8)  # first prototype channel, illustrative only

    summary = (
        f"Detection logits shape : {tuple(outputs['det'].shape)}\n"
        f"OBB params shape       : {tuple(outputs['obb'].shape)}\n"
        f"Seg. prototypes shape  : {tuple(outputs['seg_proto'].shape)}\n"
        f"Behavioral embedding   : {tuple(outputs['embed'].shape)}\n\n"
        "This demo shows raw multi-task network outputs from an "
        "(untrained, randomly initialized) DeepRodent instance for "
        "architecture inspection. Load a fine-tuned checkpoint from "
        "Hugging Face Hub to see production-quality detections, oriented "
        "boxes, and instance masks."
    )
    return proto_preview, summary


with gr.Blocks(title="DeepRodent Demo") as demo:
    gr.Markdown(
        "# 🐭 DeepRodent\n"
        "**A Robust and Generalizable Vision Framework for Automated Rodent Monitoring**\n\n"
        "Author: Teerapong Panboonyuen · "
        "[Paper & Project Page](https://kaopanboonyuen.github.io/DeepRodent) · "
        "[Code](https://github.com/kaopanboonyuen/DeepRodent) · "
        "[Weights](https://huggingface.co/kaopanboonyuen/DeepRodent)"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Upload a laboratory cage frame", type="numpy")
            btn = gr.Button("Run DeepRodent", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Segmentation prototype preview")
            out_text = gr.Textbox(label="Raw multi-task output summary", lines=8)

    btn.click(fn=run_inference, inputs=inp, outputs=[out_img, out_text])

if __name__ == "__main__":
    demo.launch()
