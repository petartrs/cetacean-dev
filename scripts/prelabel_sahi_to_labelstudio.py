#!/usr/bin/env python3
"""SAHI-tiled pre-labeling -> Label Studio pre-annotation JSON.

The gommapps (5184x3456) and MAMMALS (6480x2160) frames are far larger than the
model's 1024 training size, so a plain full-image predict shrinks tiny cetaceans
below detectability. We therefore slice each image (SAHI), run the detector per
tile, merge, and emit Label Studio tasks+predictions.

YOLO/SAHI give pixel boxes (minx,miny,maxx,maxy) in ORIGINAL image coords.
Label Studio wants percent, TOP-LEFT boxes (x,y,w,h in 0..100).

Pre-annotated boxes are tagged with the `--label` tier (default "inferred"): the
human reviewer promotes confirmed ones to "certain", deletes false ones, and adds
missed ones. Low conf (~0.15) is intentional so the reviewer mostly corrects.
"""
import argparse
import json
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="Directory of images to pre-label.")
    ap.add_argument("--out", required=True, help="Output Label Studio JSON path.")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--tile", type=int, default=640, help="SAHI slice size (px).")
    ap.add_argument("--overlap", type=float, default=0.2, help="SAHI slice overlap ratio.")
    ap.add_argument("--imgsz", type=int, default=1024, help="Per-tile inference size.")
    ap.add_argument("--label", default="inferred",
                    help="RectangleLabels value for pre-annotated boxes.")
    ap.add_argument("--ls-prefix", required=True,
                    help="LS local-files URL prefix; filename is appended. "
                         "e.g. /data/local-files/?d=gommapps-aerial-2017summer/images")
    ap.add_argument("--model-version", default="cetacean_yolo11n_sahi640")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="Debug: cap images (0=all).")
    args = ap.parse_args()

    det = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        device=args.device,
    )

    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]

    tasks = []
    total_boxes = 0
    empty = 0
    for i, img_path in enumerate(images, 1):
        result = get_sliced_prediction(
            str(img_path),
            det,
            slice_height=args.tile,
            slice_width=args.tile,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=0,
        )
        W, H = result.image_width, result.image_height
        items = []
        for op in result.object_prediction_list:
            x1, y1, x2, y2 = op.bbox.minx, op.bbox.miny, op.bbox.maxx, op.bbox.maxy
            score = float(op.score.value)
            items.append({
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": int(W),
                "original_height": int(H),
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": x1 / W * 100,
                    "y": y1 / H * 100,
                    "width": (x2 - x1) / W * 100,
                    "height": (y2 - y1) / H * 100,
                    "rectanglelabels": [args.label],
                },
                "score": score,
            })
            total_boxes += 1
        if not items:
            empty += 1
        tasks.append({
            "data": {"image": f"{args.ls_prefix}/{img_path.name}"},
            "predictions": [{
                "model_version": args.model_version,
                "score": max((r["score"] for r in items), default=0.0),
                "result": items,
            }],
        })
        if i % 25 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] {total_boxes} boxes so far", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tasks, indent=2))
    print(f"Wrote {len(tasks)} tasks ({total_boxes} boxes, {empty} empty) -> {args.out}")


if __name__ == "__main__":
    main()
