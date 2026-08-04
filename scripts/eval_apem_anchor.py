#!/usr/bin/env python3
"""APEM 151 MP real high-GSD anchor for the E5 SAHI comparison.

The synthetic altitude sweep (make_altitude_benchmark + eval_altitude_sweep) isolates
GSD on controlled pixels. This script anchors the *high-GSD real* endpoint on the one
real 151 MP APEM frame (bottlenose dolphins), the opposite extreme from the low-GSD
MAMMALS screening test. We run plain / fixed / adaptive SAHI on the real frame and
report recall / false positives / tiles / latency against a confirmed GT.

Modes:
  --mode propose : SAHI-predict at a fine tile, draw numbered candidate boxes on a
                   downsampled preview (<out>/propose_preview.jpg) and print the list.
                   A human confirms which indices are the real dolphins, then writes
                   those boxes to the GT file (data/apem/<stem>.txt, YOLO norm).
  --mode eval    : read confirmed GT (YOLO norm) and run each method (plain, fixed@S...,
                   adaptive), match to GT, write <out>/metrics.csv and a per-method
                   detections preview.

Adaptive needs a GSD. Real survey altitude for this frame is unknown, so pass --gsd
(m/px) explicitly (recommended, from the survey metadata) or --altitude with the
Phase One iXM-RS150F camera; without either, adaptive falls back to a blind sweep.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from cetacean.inference.adaptive_sahi import (
    SliceDecision, choose_slice_size, run_adaptive, _build_sahi_model,
    read_exif_geometry, gsd_from_geometry, CAMERA_REGISTRY, DEFAULT_IMGSZ,
    DEFAULT_TARGET_PX, DEFAULT_TARGET_LEN_M,
)

APEM = "phaseone_ixm_rs150f"


def yolo_to_xyxy(cx, cy, w, h, W, H):
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def load_gt(label_path: Path, W: int, H: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            boxes.append(yolo_to_xyxy(*map(float, p[1:5]), W, H))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)


def match(dets, gts, iou_thr):
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    used = [False] * len(gts)
    tp = 0
    for d in dets:
        best_j, best_i = -1, iou_thr
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou(d[:4], g)
            if v >= best_i:
                best_i, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fp = len(dets) - tp
    fn = len(gts) - sum(used)
    return tp, fp, fn


def count_tiles(W, H, S, overlap):
    from sahi.slicing import get_slice_bboxes
    return len(get_slice_bboxes(H, W, S, S, overlap, overlap))


def resolve_gsd(image_path, W, H, args):
    """GSD (m/px) priority: --gsd > --altitude+camera > EXIF altitude+camera > None."""
    if args.gsd is not None:
        return args.gsd, "user_gsd"
    cam = CAMERA_REGISTRY[APEM]
    if args.altitude is not None:
        return gsd_from_geometry(cam.pixel_pitch_um, args.altitude, cam.default_focal_mm), "user_altitude"
    ex = read_exif_geometry(image_path)
    if ex.get("altitude_m"):
        f = ex.get("focal_mm") or cam.default_focal_mm
        return gsd_from_geometry(cam.pixel_pitch_um, ex["altitude_m"], f), "exif_altitude"
    return None, "none"


def draw_preview(img, boxes, path, scale, labels=None, color=(0, 200, 0)):
    prev = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(v * scale) for v in b[:4])
        cv2.rectangle(prev, (x1, y1), (x2, y2), color, 2)
        tag = labels[i] if labels else str(i)
        cv2.putText(prev, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    cv2.imwrite(str(path), prev, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/apem/APEM_bottlenose_S15.jpg")
    ap.add_argument("--gt", default=None, help="GT label path (default <image_dir>/<stem>.txt)")
    ap.add_argument("--out", default="runs/e5/apem")
    ap.add_argument("--weights", default="runs/train/E0_s/weights/best.pt")
    ap.add_argument("--mode", choices=("propose", "eval"), default="eval")
    ap.add_argument("--fixed", type=int, nargs="+", default=[640, 1024])
    ap.add_argument("--propose-slice", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--gsd", type=float, default=None, help="ground sample distance m/px")
    ap.add_argument("--altitude", type=float, default=None, help="altitude m (iXM-RS150F)")
    ap.add_argument("--target-len", type=float, default=DEFAULT_TARGET_LEN_M)
    ap.add_argument("--target-px", type=float, default=DEFAULT_TARGET_PX)
    ap.add_argument("--min-slice", type=int, default=512)
    ap.add_argument("--preview-scale", type=float, default=0.08)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    img_path = Path(args.image).resolve()
    gt_path = Path(args.gt) if args.gt else img_path.with_suffix(".txt")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import cv2 as _cv2
    _cv2.setNumThreads(0)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    print(f"APEM {W}x{H} ({W*H/1e6:.1f} MP)")

    if args.mode == "propose":
        model = _build_sahi_model(args.weights, args.conf, args.imgsz, args.device)
        dec = SliceDecision(args.propose_slice, "sahi", None, None, None, "propose",
                            f"fixed {args.propose_slice}")
        t0 = time.perf_counter()
        dets = run_adaptive(img_path, args.weights, dec, conf=args.conf,
                            overlap=args.overlap, imgsz=args.imgsz, device=args.device,
                            sahi_model=model)
        dt = time.perf_counter() - t0
        dets = sorted(dets, key=lambda d: d[4], reverse=True)
        print(f"proposed {len(dets)} candidates in {dt:.1f}s (slice {args.propose_slice}):")
        for i, d in enumerate(dets):
            x1, y1, x2, y2, s = d
            print(f"  [{i:>2}] conf={s:.3f} xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"size={max(x2-x1, y2-y1):.0f}px")
        draw_preview(img, dets, out / "propose_preview.jpg", args.preview_scale,
                     labels=[f"{i}:{d[4]:.2f}" for i, d in enumerate(dets)])
        # dump raw candidate boxes (xyxy + conf) for the confirm step
        with open(out / "propose_candidates.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["idx", "x1", "y1", "x2", "y2", "conf"])
            for i, d in enumerate(dets):
                wr.writerow([i] + [round(v, 1) for v in d[:4]] + [round(d[4], 4)])
        print(f"\npreview -> {out/'propose_preview.jpg'} ; candidates -> "
              f"{out/'propose_candidates.csv'}")
        print("Confirm the real dolphins, then write their YOLO-norm boxes to", gt_path)
        return

    # eval mode
    gts = load_gt(gt_path, W, H)
    if not gts:
        raise SystemExit(f"no GT at {gt_path}; run --mode propose first and write GT")
    print(f"GT dolphins: {len(gts)}")
    gsd, gsd_src = resolve_gsd(img_path, W, H, args)
    print(f"GSD: {gsd*100:.3f} cm/px (source={gsd_src})" if gsd else "GSD: none (blind)")

    methods = ["plain"] + [f"fixed{S}" for S in args.fixed] + ["adaptive"]
    sahi_model = _build_sahi_model(args.weights, args.conf, args.imgsz, args.device)
    from ultralytics import YOLO
    yolo_model = YOLO(args.weights)

    results = []
    for meth in methods:
        if meth == "plain":
            dec = SliceDecision(None, "plain", gsd, None, None, "eval", "plain")
            tiles = 1
        elif meth == "adaptive":
            if gsd is None:
                print("  [adaptive] no GSD -> skipping (supply --gsd/--altitude)")
                continue
            dec = choose_slice_size(W, H, gsd=gsd, target_len_m=args.target_len,
                                    target_px=args.target_px, imgsz=args.imgsz,
                                    min_slice=args.min_slice)
            tiles = 1 if dec.mode == "plain" else count_tiles(W, H, dec.slice_size, args.overlap)
        else:
            S = int(meth.replace("fixed", ""))
            dec = SliceDecision(S, "sahi", gsd, None, None, "eval", f"fixed {S}")
            tiles = count_tiles(W, H, S, args.overlap)

        t0 = time.perf_counter()
        dets = run_adaptive(img_path, args.weights, dec, conf=args.conf,
                            overlap=args.overlap, imgsz=args.imgsz, device=args.device,
                            yolo_model=yolo_model, sahi_model=sahi_model)
        secs = time.perf_counter() - t0
        tp, fp, fn = match(dets, gts, args.iou)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        results.append({
            "method": meth, "slice": dec.slice_size or 0, "tiles": tiles,
            "n_det": len(dets), "tp": tp, "fp": fp, "fn": fn,
            "recall": round(rec, 4), "sec": round(secs, 2),
        })
        print(f"  {meth:<10} slice={dec.slice_size or 0:<5} tiles={tiles:<5} "
              f"det={len(dets):<4} TP={tp} FP={fp} FN={fn} R={rec:.2f} ({secs:.1f}s)")
        draw_preview(img, dets, out / f"det_{meth}.jpg", args.preview_scale,
                     labels=[f"{d[4]:.2f}" for d in dets], color=(0, 165, 255))

    with open(out / "metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)
    print(f"\nmetrics -> {out/'metrics.csv'}")


if __name__ == "__main__":
    main()
