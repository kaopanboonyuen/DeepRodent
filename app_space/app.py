"""
🐭 DeepRodent
A Robust and Generalizable Vision Framework for Automated Rodent Monitoring

Gradio inference UI for the DeepRodent DeepRodent-Seg model.
Deployed on Hugging Face Spaces (ZeroGPU-compatible).

Model summary (auto-detected from DeepRodent_WEIGHT.pt):
    Architecture : DeepRodent-Seg (instance segmentation)
    Task         : segment
    Classes      : 1  ->  {0: "rodent"}
    Input size   : 640 x 640

Weight source:
    https://huggingface.co/kaopanboonyuen/DeepRodent

Run locally:
    pip install -r requirements.txt
    python app.py
"""

# `spaces` must be imported before torch-using libraries so its CUDA
# emulation patch is in place when torch/ultralytics initialize.
import spaces

import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_REPO = "kaopanboonyuen/DeepRodent"
MODEL_FILENAME = "DeepRodent_WEIGHT.pt"
IMG_SIZE = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

BOX_COLOR = (175, 82, 222)        # Apple "purple" accent, RGB
MASK_COLOR = (94, 92, 230)        # Apple "indigo" accent, RGB
TEXT_COLOR = (255, 255, 255)

# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------
print(f"Downloading weights from {MODEL_REPO}...")
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

print("Loading DeepRodent model...")
model = YOLO(MODEL_PATH)

# Per ZeroGPU docs: models must be moved to 'cuda' at module level (not
# inside a @spaces.GPU function) for efficient placement. This is a no-op
# on CPU-only hardware / local runs without a GPU.
try:
    model.to("cuda")
except Exception as e:
    print(f"Running on CPU (no CUDA device available): {e}")

CLASS_NAMES = model.names
print(f"Model loaded. Task={model.task} | Classes={CLASS_NAMES}")


# --------------------------------------------------------------------------
# Inference + drawing
# --------------------------------------------------------------------------
def draw_results(image_rgb: np.ndarray, result, conf_thres: float, show_masks: bool, show_labels: bool):
    """Draw segmentation masks + boxes + labels on top of the image."""
    canvas = image_rgb.copy()
    overlay = canvas.copy()

    boxes = result.boxes
    masks = result.masks

    count = 0
    if boxes is None or len(boxes) == 0:
        return canvas, 0

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        if conf < conf_thres:
            continue
        count += 1

        cls_id = int(boxes.cls[i])
        label = CLASS_NAMES.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

        # Mask fill
        if show_masks and masks is not None:
            mask = masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]))
            mask_bool = mask > 0.5
            overlay[mask_bool] = MASK_COLOR

        # Box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2, lineType=cv2.LINE_AA)

        # Label
        if show_labels:
            tag = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty1 = max(y1 - th - 10, 0)
            cv2.rectangle(canvas, (x1, ty1), (x1 + tw + 10, ty1 + th + 10), BOX_COLOR, -1)
            cv2.putText(
                canvas, tag, (x1 + 5, ty1 + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, lineType=cv2.LINE_AA
            )

    if show_masks:
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
        # redraw boxes/labels crisply on top of blended mask
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2, lineType=cv2.LINE_AA)
            if show_labels:
                cls_id = int(boxes.cls[i])
                label = CLASS_NAMES.get(cls_id, str(cls_id))
                tag = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                ty1 = max(y1 - th - 10, 0)
                cv2.rectangle(canvas, (x1, ty1), (x1 + tw + 10, ty1 + th + 10), BOX_COLOR, -1)
                cv2.putText(
                    canvas, tag, (x1 + 5, ty1 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, lineType=cv2.LINE_AA
                )

    return canvas, count


@spaces.GPU(duration=30)
def run_inference(image, conf_thres, iou_thres, show_masks, show_labels):
    if image is None:
        return None, "_Upload or select an image to begin._"

    t0 = time.time()
    image_rgb = np.array(image.convert("RGB"))

    results = model.predict(
        source=image_rgb,
        imgsz=IMG_SIZE,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False,
    )
    result = results[0]

    annotated, count = draw_results(image_rgb, result, conf_thres, show_masks, show_labels)
    elapsed_ms = (time.time() - t0) * 1000

    summary = (
        f"### 🐭 Detection Summary\n"
        f"- **Rodents detected:** `{count}`\n"
        f"- **Confidence threshold:** `{conf_thres:.2f}`\n"
        f"- **IoU threshold:** `{iou_thres:.2f}`\n"
        f"- **Inference time:** `{elapsed_ms:.1f} ms`\n"
        f"- **Input size:** `{IMG_SIZE}×{IMG_SIZE}`"
    )

    return Image.fromarray(annotated), summary


def _batch_duration(files, conf_thres, iou_thres, show_masks, show_labels):
    # ~8s per image is a safe upper bound for a nano seg model at 640px;
    # keeps queue priority high for small batches while covering large ones.
    n = len(files) if files else 1
    return min(max(n * 8, 15), 170)


@spaces.GPU(duration=_batch_duration)
def run_batch(files, conf_thres, iou_thres, show_masks, show_labels):
    if not files:
        return [], "_No images uploaded._"

    gallery_items = []
    total_count = 0
    for f in files:
        img = Image.open(f.name).convert("RGB")
        image_rgb = np.array(img)
        results = model.predict(
            source=image_rgb, imgsz=IMG_SIZE, conf=conf_thres, iou=iou_thres, verbose=False
        )
        annotated, count = draw_results(image_rgb, results[0], conf_thres, show_masks, show_labels)
        total_count += count
        gallery_items.append((Image.fromarray(annotated), f"{Path(f.name).name} — {count} rodent(s)"))

    summary = (
        f"### 📦 Batch Summary\n"
        f"- **Images processed:** `{len(files)}`\n"
        f"- **Total rodents detected:** `{total_count}`"
    )
    return gallery_items, summary


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --dr-bg: #f5f5f7;
    --dr-panel: #ffffff;
    --dr-border: #e5e5ea;
    --dr-text: #1d1d1f;
    --dr-subtext: #86868b;
    --dr-accent1: #ff375f;
    --dr-accent2: #af52de;
    --dr-accent3: #5e5ce6;
    --dr-accent4: #007aff;
}

* {
    font-family: -apple-system, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif !important;
}

.gradio-container {
    background: linear-gradient(180deg, #fbfbfd 0%, #f5f5f7 100%) !important;
    color: var(--dr-text) !important;
}

/* ---------- Header ---------- */
#dr-header {
    position: relative;
    text-align: center;
    padding: 48px 16px 26px 16px;
    overflow: hidden;
}
#dr-header::before {
    content: "";
    position: absolute;
    top: -120px;
    left: 50%;
    transform: translateX(-50%);
    width: 620px;
    height: 320px;
    background: radial-gradient(circle at 30% 40%, rgba(255,55,95,0.16), transparent 60%),
                radial-gradient(circle at 65% 55%, rgba(94,92,230,0.16), transparent 60%),
                radial-gradient(circle at 50% 65%, rgba(0,122,255,0.14), transparent 60%);
    filter: blur(30px);
    pointer-events: none;
    z-index: 0;
    animation: dr-glow-drift 10s ease-in-out infinite alternate;
}
@keyframes dr-glow-drift {
    0%   { transform: translateX(-52%) scale(1); }
    100% { transform: translateX(-48%) scale(1.08); }
}
#dr-header h1, #dr-header p, #dr-header .dr-badge {
    position: relative;
    z-index: 1;
}
#dr-header h1 {
    font-size: 2.7rem;
    margin-bottom: 6px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #1d1d1f !important;
    background-image: linear-gradient(100deg, var(--dr-accent1) 0%, var(--dr-accent2) 35%, var(--dr-accent3) 65%, var(--dr-accent4) 100%) !important;
    background-size: 200% auto !important;
    background-clip: text !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: dr-shimmer 6s ease-in-out infinite;
}
@keyframes dr-shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@supports not ((background-clip: text) or (-webkit-background-clip: text)) {
    #dr-header h1 {
        -webkit-text-fill-color: initial !important;
        color: #1d1d1f !important;
        background: none !important;
    }
}
#dr-header p {
    color: var(--dr-subtext);
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0;
}
#dr-header p.dr-author {
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    color: #a1a1a6;
    margin: 4px 0 16px 0;
    text-transform: uppercase;
}

/* ---------- Badges ---------- */
.dr-badge {
    display: inline-block;
    background: rgba(0, 0, 0, 0.035);
    color: #48484a;
    border: 1px solid var(--dr-border);
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    margin: 3px 5px;
    backdrop-filter: blur(6px);
}

/* ---------- Panels / Cards ---------- */
.dr-panel {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    border-radius: 20px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 12px 32px rgba(0,0,0,0.06) !important;
}

/* ---------- Tabs ---------- */
.tab-nav, [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--dr-border) !important;
}
button[role="tab"] {
    color: var(--dr-subtext) !important;
    font-weight: 500 !important;
    border-radius: 10px 10px 0 0 !important;
}
button[role="tab"].selected, button[role="tab"][aria-selected="true"] {
    color: var(--dr-text) !important;
    font-weight: 600 !important;
}

/* ---------- Buttons ---------- */
button.primary, .primary {
    background: linear-gradient(100deg, var(--dr-accent1), var(--dr-accent3)) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(94, 92, 230, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(94, 92, 230, 0.32) !important;
}

/* ---------- Inputs / sliders / accordions ---------- */
input, textarea, .gr-box, .block {
    border-radius: 14px !important;
}
.gr-box, .wrap, .container {
    border-color: var(--dr-border) !important;
}
label, .label-wrap span {
    color: var(--dr-text) !important;
    font-weight: 500 !important;
}
.gradio-container input[type="range"] {
    accent-color: var(--dr-accent3) !important;
}

/* ---------- Markdown / tables ---------- */
.prose table {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--dr-border) !important;
}
.prose th {
    background: #f5f5f7 !important;
    color: var(--dr-text) !important;
}
.prose code {
    background: rgba(0,0,0,0.045) !important;
    color: #af52de !important;
    border-radius: 6px;
}

footer {visibility: hidden}
"""

with gr.Blocks(
    title="DeepRodent — Automated Rodent Monitoring",
) as demo:

    gr.HTML(
        """
        <div id="dr-header">
            <h1>🐭 DeepRodent</h1>
            <p>A Robust and Generalizable Vision Framework for Automated Rodent Monitoring</p>
            <p class="dr-author">Author: Teerapong Panboonyuen</p>
            <span class="dr-badge">DeepRodent-Seg</span>
            <span class="dr-badge">Instance Segmentation</span>
            <span class="dr-badge">1 class · rodent</span>
        </div>
        """
    )

    with gr.Tabs():
        # ---------------- Single image tab ----------------
        with gr.Tab("🔍 Single Image"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="dr-panel"):
                    input_image = gr.Image(type="pil", label="Input Image", height=340)

                    with gr.Accordion("⚙️ Inference Settings", open=True):
                        conf_slider = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence Threshold")
                        iou_slider = gr.Slider(0.05, 0.95, value=DEFAULT_IOU, step=0.05, label="IoU Threshold (NMS)")
                        with gr.Row():
                            show_masks_cb = gr.Checkbox(value=True, label="Show segmentation masks")
                            show_labels_cb = gr.Checkbox(value=True, label="Show labels & scores")

                    run_btn = gr.Button("▶ Run Detection", variant="primary", size="lg")

                with gr.Column(scale=1, elem_classes="dr-panel"):
                    output_image = gr.Image(type="pil", label="Detection Result", height=340)
                    summary_md = gr.Markdown("_Run detection to see results here._")

            _sample_dir = Path("rodent_sample_images")
            _sample_paths = [
                str(_sample_dir / f"rodent_sample_images_{i:02d}.jpg") for i in (1, 2, 3)
            ]
            _existing_samples = [[p] for p in _sample_paths if Path(p).exists()]
            if _existing_samples:
                gr.Examples(
                    examples=_existing_samples,
                    inputs=input_image,
                    label="Sample Images (from rodent_sample_images/)",
                )

            run_btn.click(
                fn=run_inference,
                inputs=[input_image, conf_slider, iou_slider, show_masks_cb, show_labels_cb],
                outputs=[output_image, summary_md],
            )

        # ---------------- Batch tab ----------------
        with gr.Tab("📦 Batch Processing"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="dr-panel"):
                    batch_files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload multiple images",
                    )
                    with gr.Accordion("⚙️ Inference Settings", open=True):
                        batch_conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence Threshold")
                        batch_iou = gr.Slider(0.05, 0.95, value=DEFAULT_IOU, step=0.05, label="IoU Threshold (NMS)")
                        with gr.Row():
                            batch_masks_cb = gr.Checkbox(value=True, label="Show segmentation masks")
                            batch_labels_cb = gr.Checkbox(value=True, label="Show labels & scores")
                    batch_btn = gr.Button("▶ Run Batch Detection", variant="primary", size="lg")

                with gr.Column(scale=2, elem_classes="dr-panel"):
                    batch_gallery = gr.Gallery(label="Results", columns=3, height=460, object_fit="contain")
                    batch_summary_md = gr.Markdown("_Upload images and run detection._")

            batch_btn.click(
                fn=run_batch,
                inputs=[batch_files, batch_conf, batch_iou, batch_masks_cb, batch_labels_cb],
                outputs=[batch_gallery, batch_summary_md],
            )

        # ---------------- About tab ----------------
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                f"""
                ## Model Card

                | Field | Value |
                |---|---|
                | **Project** | DeepRodent |
                | **Author** | Kao |
                | **Architecture** | DeepRodent-Seg (Ultralytics) |
                | **Task** | Instance Segmentation |
                | **Classes** | `{dict(CLASS_NAMES)}` |
                | **Input resolution** | {IMG_SIZE} × {IMG_SIZE} |
                | **Weight file** | [`{MODEL_REPO}`](https://huggingface.co/{MODEL_REPO}) / `{MODEL_FILENAME}` |

                DeepRodent detects and segments rodents in images for automated
                monitoring pipelines — e.g. lab facility surveillance, pest tracking,
                or behavioral studies — producing both bounding boxes and
                pixel-level masks per detected rodent.

                **Tips for your team demo:**
                - Lower the confidence threshold to catch more (but noisier) detections.
                - Raise IoU threshold if overlapping rodents are being merged into one box.
                - Toggle masks off for a cleaner "box-only" view during presentations.
                """
            )

    gr.HTML(
        "<div style='text-align:center; color:#5b6270; padding: 10px 0 4px 0; font-size:0.85rem;'>"
        "DeepRodent · Vision framework for automated rodent monitoring · Author: Kao"
        "</div>"
    )

if __name__ == "__main__":
    # theme/css passed here for compatibility across Gradio versions
    # (Gradio >=6 expects them in launch(), older versions accept them
    # here too via the fallback below)
    apple_theme = gr.themes.Base(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="gray",
        font=[
            gr.themes.Font("-apple-system"),
            gr.themes.Font("BlinkMacSystemFont"),
            gr.themes.Font("Segoe UI"),
            gr.themes.Font("sans-serif"),
        ],
    ).set(
        body_background_fill="#f5f5f7",
        block_background_fill="#ffffff",
        block_border_color="#e5e5ea",
        block_radius="20px",
    )
    try:
        demo.launch(theme=apple_theme, css=CUSTOM_CSS)
    except TypeError:
        demo.launch()
