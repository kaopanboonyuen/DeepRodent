# 🐭 DeepRodent — Inference UI

A Robust and Generalizable Vision Framework for Automated Rodent Monitoring

## Model info (auto-detected from DeepRodent_WEIGHT.pt)
- Architecture: YOLOv8n-seg (Ultralytics)
- Task: instance segmentation
- Classes: 1 -> {0: "rodent"}
- Input size: 640x640

## Setup
```bash
pip install -r requirements.txt
```

Folder layout expected (already set up for you):
```
deeprodent/
├── app.py
├── DeepRodent_WEIGHT.pt
├── requirements.txt
└── rodent_sample_images/
    ├── rodent_sample_images_01.jpg
    ├── rodent_sample_images_02.jpg
    └── rodent_sample_images_03.jpg   <- add your 3 sample images here
```

> Note: your 3 sample images weren't included in the upload, so this folder
> is currently empty. Drop your 3 files in with the exact names above and
> the "Examples" gallery in the Single Image tab will pick them up
> automatically. If you skip this, the app still works perfectly —
> just drag-and-drop any image manually.

## Run
```bash
python app.py
```
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

## Features
- **Single Image tab**: upload one image, adjust confidence/IoU thresholds,
  toggle masks/labels, see live inference time.
- **Batch Processing tab**: upload multiple images at once, get a gallery
  of annotated results + aggregate rodent count.
- **About tab**: model card for quick reference in your team demo.
