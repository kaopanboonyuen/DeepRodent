"""
🐭 DeepRodent
A Robust and Generalizable Vision Framework for Automated Rodent Monitoring

Gradio inference UI for the DeepRodent YOLOv8n-seg model.

Model summary (auto-detected from DeepRodent_WEIGHT.pt):
    Architecture : YOLOv8n-seg (instance segmentation)
    Task         : segment
    Classes      : 1  ->  {0: "rodent"}
    Input size   : 640 x 640

Run:
    pip install gradio ultralytics opencv-python-headless
    python app.py
"""

import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "DeepRodent_WEIGHT.pt"          # place next to this app.py
IMG_SIZE = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

BOX_COLOR = (255, 61, 90)        # BGR-agnostic, used as RGB below
MASK_COLOR = (255, 61, 90)
TEXT_COLOR = (255, 255, 255)

# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------
print("Loading DeepRodent model...")
model = YOLO(MODEL_PATH)
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
    --dr-bg: #0f1117;
    --dr-panel: #161a23;
    --dr-accent: #ff3d5a;
    --dr-accent-soft: rgba(255, 61, 90, 0.12);
    --dr-text: #eaecef;
    --dr-subtext: #9aa1ad;
}
.gradio-container {
    background: radial-gradient(1200px 600px at 10% -10%, #1a1f2b 0%, var(--dr-bg) 55%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
#dr-header {
    text-align: center;
    padding: 28px 16px 18px 16px;
}
#dr-header h1 {
    font-size: 2.4rem;
    margin-bottom: 4px;
    background: linear-gradient(90deg, #ff3d5a, #ff9a5a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
#dr-header p {
    color: var(--dr-subtext);
    font-size: 1.02rem;
    margin-top: 0;
}
.dr-badge {
    display: inline-block;
    background: var(--dr-accent-soft);
    color: var(--dr-accent);
    border: 1px solid rgba(255, 61, 90, 0.35);
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin: 2px 4px;
}
.dr-panel {
    background: var(--dr-panel) !important;
    border: 1px solid #232838 !important;
    border-radius: 16px !important;
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
            <span class="dr-badge">YOLOv8n-seg</span>
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

            gr.Examples(
                examples=[
                    ["rodent_sample_images/rodent_sample_images_01.jpg"],
                    ["rodent_sample_images/rodent_sample_images_02.jpg"],
                    ["rodent_sample_images/rodent_sample_images_03.jpg"],
                ],
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
                | **Architecture** | YOLOv8n-seg (Ultralytics) |
                | **Task** | Instance Segmentation |
                | **Classes** | `{dict(CLASS_NAMES)}` |
                | **Input resolution** | {IMG_SIZE} × {IMG_SIZE} |
                | **Weight file** | `{MODEL_PATH}` |

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
        "DeepRodent · Vision framework for automated rodent monitoring"
        "</div>"
    )

if __name__ == "__main__":
    # theme/css passed here for compatibility across Gradio versions
    # (Gradio >=6 expects them in launch(), older versions accept them
    # here too via the fallback below)
    try:
        demo.launch(theme=gr.themes.Soft(primary_hue="rose", neutral_hue="slate"), css=CUSTOM_CSS)
    except TypeError:
        demo.launch()
