#!/usr/bin/env python3
"""Draw ground-truth YOLO boxes on images for quick visual review (no model).

Usage:
    python scripts/draw_gt.py --images data/normalized/dryad/images \
        --labels data/normalized/dryad/labels --out runs/viz/dryad_gt
"""
from __future__ import annotations

import argparse
import os

import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
GT_COLOR = (0, 200, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None, help="defaults to images dir with 'images'->'labels'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    labels = args.labels or args.images.replace("images", "labels")
    os.makedirs(args.out, exist_ok=True)

    files = sorted(f for f in os.listdir(args.images) if f.endswith(IMG_EXTS))
    if args.limit:
        files = files[: args.limit]

    n_img = n_box = 0
    for fn in files:
        ip = os.path.join(args.images, fn)
        lp = os.path.join(labels, os.path.splitext(fn)[0] + ".txt")
        img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = 0
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, p[:5])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, max(2, w // 400))
                boxes += 1
        cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(img, f"{fn}  GT={boxes}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(args.out, fn), img)
        n_img += 1
        n_box += boxes

    print(f"images: {n_img}  GT boxes: {n_box}  -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
