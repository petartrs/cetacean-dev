#!/usr/bin/env python3
"""Run a YOLO model over a folder of images and emit a Label Studio
pre-annotations JSON file (tasks + predictions) for model-assisted labeling.

YOLO gives normalized, CENTER-based boxes (cx, cy, w, h in 0..1).
Label Studio wants percent, TOP-LEFT-based boxes (x, y, w, h in 0..100).
"""
import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}

DEFAULT_MODEL = os.environ.get("CETACEAN_WEIGHTS", "models/cetacean_yolo11n.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--images", default="data/assist_pool/images")
    ap.add_argument("--out", default="data/assist_pool/predictions.json")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold. Lower = more (weaker) suggestions to correct.")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="Inference size; match training (1024).")
    ap.add_argument("--label", default="cetacean",
                    help="Class label; must match the RectangleLabels <Label value=...> in the LS project.")
    ap.add_argument("--ls-prefix", default="/data/local-files/?d=assist_pool/images",
                    help="Label Studio local-files URL prefix (relative to DOCUMENT_ROOT).")
    ap.add_argument("--model-version", default="stage1_n_old")
    args = ap.parse_args()

    model = YOLO(args.model)
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)

    tasks = []
    total_boxes = 0
    empty = 0
    for img_path in images:
        res = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        height, width = res.orig_shape  # (H, W)

        result_items = []
        for box in res.boxes:
            cx, cy, w, h = box.xywhn[0].tolist()  # normalized center-based 0..1
            score = float(box.conf[0])
            result_items.append({
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": int(width),
                "original_height": int(height),
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": (cx - w / 2) * 100,   # top-left x, percent
                    "y": (cy - h / 2) * 100,   # top-left y, percent
                    "width": w * 100,
                    "height": h * 100,
                    "rectanglelabels": [args.label],
                },
                "score": score,
            })
            total_boxes += 1

        if not result_items:
            empty += 1

        tasks.append({
            "data": {"image": f"{args.ls_prefix}/{img_path.name}"},
            "predictions": [{
                "model_version": args.model_version,
                "score": max((r["score"] for r in result_items), default=0.0),
                "result": result_items,
            }],
        })

    Path(args.out).write_text(json.dumps(tasks, indent=2))
    print(f"Wrote {len(tasks)} tasks ({total_boxes} boxes, {empty} with no prediction) -> {args.out}")


if __name__ == "__main__":
    main()
