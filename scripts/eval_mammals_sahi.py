#!/usr/bin/env python3
"""SAHI-tiled eval of a model on the labeled MAMMALS screening test set.

Reports box-level precision/recall (IoU>=0.3, small-object appropriate),
frame-level hit rate (screening), and false-positives-per-image, over a
confidence sweep. Locked SAHI: tile 640 / overlap 0.2 / imgsz 1024.
"""
import argparse
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_gt(label_path, W, H):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:5])
        boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                      (cx + w / 2) * W, (cy + h / 2) * H])
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--base-conf", type=float, default=0.10,
                    help="Detector conf floor; sweep thresholds applied post-hoc.")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--confs", default="0.10,0.25,0.40,0.50")
    args = ap.parse_args()

    det = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.model,
        confidence_threshold=args.base_conf, image_size=args.imgsz,
        device=args.device,
    )
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    lbl_dir = Path(args.labels)

    # Collect predictions once at the conf floor, then threshold in-memory.
    per_img = []  # (n_gt, [(score, box), ...])
    n_pos_frames = n_neg_frames = 0
    for i, img in enumerate(images, 1):
        res = get_sliced_prediction(
            str(img), det, slice_height=args.tile, slice_width=args.tile,
            overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
            verbose=0,
        )
        W, H = res.image_width, res.image_height
        gt = load_gt(lbl_dir / f"{img.stem}.txt", W, H)
        preds = [(float(op.score.value),
                  [op.bbox.minx, op.bbox.miny, op.bbox.maxx, op.bbox.maxy])
                 for op in res.object_prediction_list]
        per_img.append((gt, preds))
        if gt:
            n_pos_frames += 1
        else:
            n_neg_frames += 1
        if i % 20 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] predicted", flush=True)

    total_gt = sum(len(gt) for gt, _ in per_img)
    print(f"\nImages: {len(images)} ({n_pos_frames} with GT, {n_neg_frames} empty/neg)")
    print(f"Total GT boxes: {total_gt}")
    print(f"\nSAHI tile={args.tile} overlap={args.overlap} imgsz={args.imgsz} IoU>={args.iou}")
    print(f"{'conf':>6} | {'boxP':>6} {'boxR':>6} {'F1':>6} | "
          f"{'TP':>4} {'FP':>4} {'FN':>4} | {'frameR':>7} {'FPPI':>6}")
    print("-" * 74)
    for conf in [float(c) for c in args.confs.split(",")]:
        TP = FP = FN = 0
        frames_hit = 0
        fp_frames = 0
        for gt, preds in per_img:
            dets = sorted([p for p in preds if p[0] >= conf], key=lambda x: -x[0])
            matched = set()
            tp_here = 0
            for score, box in dets:
                best_j, best_iou = -1, args.iou
                for j, g in enumerate(gt):
                    if j in matched:
                        continue
                    ov = iou(box, g)
                    if ov >= best_iou:
                        best_iou, best_j = ov, j
                if best_j >= 0:
                    matched.add(best_j)
                    tp_here += 1
                else:
                    FP += 1
            TP += tp_here
            FN += len(gt) - len(matched)
            if gt and matched:
                frames_hit += 1
            if len(dets) - tp_here > 0:
                fp_frames += 1
        boxP = TP / (TP + FP) if TP + FP else 0.0
        boxR = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * boxP * boxR / (boxP + boxR) if boxP + boxR else 0.0
        frameR = frames_hit / n_pos_frames if n_pos_frames else 0.0
        fppi = FP / len(images)
        print(f"{conf:>6.2f} | {boxP:>6.3f} {boxR:>6.3f} {f1:>6.3f} | "
              f"{TP:>4} {FP:>4} {FN:>4} | {frameR:>7.3f} {fppi:>6.3f}")


if __name__ == "__main__":
    main()
