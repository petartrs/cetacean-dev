#!/usr/bin/env python3
"""Evaluate plain vs fixed-slice SAHI vs adaptive SAHI on the altitude-sweep benchmark.

For every synthetic survey frame (one simulated altitude), we run:
    * plain          — whole-frame inference at imgsz (the naive baseline).
    * fixed@S         — SAHI with a fixed crop size S (one per configured slice) — the
                        "tuned for one altitude" baseline that is wrong everywhere else.
    * adaptive        — SAHI whose crop size is chosen per-frame from the frame's GSD
                        (the paper's contribution), via adaptive_sahi.choose_slice_size.

Each method's detections are matched to the exact synthetic GT → recall / precision / F1.
We also log tiles-per-frame (compute cost) and wall-clock latency. Outputs:
    <bench>/eval/metrics.csv
    <bench>/eval/recall_vs_altitude.png     (C3 — core quantitative result)
    <bench>/eval/f1_vs_altitude.png
    <bench>/eval/tiles_vs_altitude.png       (C4 — efficiency: adaptive spends tiles only
                                              when the geometry says it must)

Usage:
    python scripts/eval_altitude_sweep.py --bench data/benchmark/altitude_sweep \
        --fixed 1024 2048 --conf 0.3
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from cetacean.inference.adaptive_sahi import (
    SliceDecision, choose_slice_size, run_adaptive, _build_sahi_model,
    DEFAULT_WEIGHTS, DEFAULT_TARGET_PX, DEFAULT_TARGET_LEN_M, DEFAULT_IMGSZ,
)


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
    """Greedy IoU matching (highest-score det first). Returns (tp, fp, fn)."""
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


def prf(tp, fp, fn):
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return rec, prec, f1


def count_tiles(W, H, S, overlap):
    from sahi.slicing import get_slice_bboxes
    return len(get_slice_bboxes(H, W, S, S, overlap, overlap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark/altitude_sweep")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--fixed", type=int, nargs="+", default=[1024, 2048],
                    help="fixed SAHI crop sizes to compare against")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--target-len", type=float, default=DEFAULT_TARGET_LEN_M)
    ap.add_argument("--target-px", type=float, default=DEFAULT_TARGET_PX)
    ap.add_argument("--skip-px", type=float, default=None,
                    help="detection-floor apparent size for the skip-if-large decision "
                         "(defaults to --target-px)")
    ap.add_argument("--min-slice", type=int, default=512,
                    help="floor on adaptive crop size (avoid the over-magnify FP-flood zone)")
    ap.add_argument("--max-len", type=float, default=None,
                    help="largest expected animal length (m) for adaptive_range; "
                         "defaults to per-frame len_max_m from the manifest")
    ap.add_argument("--max-fill", type=float, default=1.0,
                    help="fraction of a tile the largest animal may fill in adaptive_range")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None,
                    help="output dir for metrics/plots (default <bench>/eval)")
    args = ap.parse_args()

    bench = Path(args.bench).resolve()
    out = Path(args.out).resolve() if args.out else bench / "eval"
    out.mkdir(parents=True, exist_ok=True)

    # manifest → frames + altitude + GSD
    rows = list(csv.DictReader(open(bench / "manifest.csv")))
    if not rows:
        raise SystemExit("empty manifest")

    from ultralytics import YOLO
    print("loading models...")
    yolo_model = YOLO(args.weights)
    sahi_model = _build_sahi_model(args.weights, args.conf, args.imgsz, args.device)

    methods = ["plain"] + [f"fixed{S}" for S in args.fixed] + ["adaptive", "adaptive_range"]
    results = []  # dicts: method, altitude_m, gsd_cm_px, recall, prec, f1, tiles, ms, slice

    for r in rows:
        frame = bench / r["frame"]
        alt = float(r["altitude_m"])
        gsd_m = float(r["gsd_cm_px"]) / 100.0
        img = cv2.imread(str(frame))
        H, W = img.shape[:2]
        gts = load_gt(bench / "labels" / f"{frame.stem}.txt", W, H)
        print(f"\nalt {alt:>4.0f}m  gsd {r['gsd_cm_px']}cm/px  GT {len(gts)}")

        for meth in methods:
            if meth == "plain":
                dec = SliceDecision(None, "plain", gsd_m, None, None, "eval", "plain")
                tiles = 1
            elif meth == "adaptive":
                dec = choose_slice_size(
                    W, H, gsd=gsd_m, target_len_m=args.target_len,
                    target_px=args.target_px, skip_px=args.skip_px,
                    imgsz=args.imgsz, min_slice=args.min_slice)
                tiles = 1 if dec.mode == "plain" else count_tiles(W, H, dec.slice_size, args.overlap)
            elif meth == "adaptive_range":
                max_len = args.max_len or (float(r["len_max_m"]) if r.get("len_max_m")
                                           else args.target_len)
                dec = choose_slice_size(
                    W, H, gsd=gsd_m, target_len_m=args.target_len,
                    target_px=args.target_px, skip_px=args.skip_px,
                    imgsz=args.imgsz, min_slice=args.min_slice,
                    max_len_m=max_len, max_fill=args.max_fill)
                tiles = 1 if dec.mode == "plain" else count_tiles(W, H, dec.slice_size, args.overlap)
            else:  # fixed{S}
                S = int(meth.replace("fixed", ""))
                dec = SliceDecision(S, "sahi", gsd_m, None, None, "eval", f"fixed {S}")
                tiles = count_tiles(W, H, S, args.overlap)

            t0 = time.perf_counter()
            dets = run_adaptive(frame, args.weights, dec, conf=args.conf,
                                overlap=args.overlap, imgsz=args.imgsz, device=args.device,
                                yolo_model=yolo_model, sahi_model=sahi_model)
            ms = (time.perf_counter() - t0) * 1000
            tp, fp, fn = match(dets, gts, args.iou)
            rec, prec, f1 = prf(tp, fp, fn)
            results.append({
                "method": meth, "altitude_m": alt, "gsd_cm_px": float(r["gsd_cm_px"]),
                "recall": round(rec, 4), "precision": round(prec, 4), "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn, "tiles": tiles, "ms": round(ms, 1),
                "slice": dec.slice_size if dec.slice_size else 0,
            })
            print(f"  {meth:<11} slice={dec.slice_size or 0:<5} tiles={tiles:<4} "
                  f"R={rec:.2f} P={prec:.2f} F1={f1:.2f} ({ms:.0f}ms)")

    # CSV
    with open(out / "metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        alts = sorted({row["altitude_m"] for row in results})

        def series(meth, key):
            return [next(x[key] for x in results if x["method"] == meth and x["altitude_m"] == a)
                    for a in alts]

        for key, fname, ylab in [("recall", "recall_vs_altitude.png", "Recall"),
                                 ("f1", "f1_vs_altitude.png", "F1")]:
            plt.figure(figsize=(8, 5))
            for meth in methods:
                style = "-o" if meth == "adaptive" else "--s"
                lw = 2.6 if meth == "adaptive" else 1.5
                plt.plot(alts, series(meth, key), style, label=meth, linewidth=lw)
            plt.xlabel("Simulated altitude (m)  —  higher = smaller animals")
            plt.ylabel(ylab)
            plt.title(f"{ylab} vs altitude: plain / fixed-slice / adaptive SAHI")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / fname, dpi=130)
            plt.close()

        plt.figure(figsize=(8, 5))
        for meth in methods:
            if meth == "plain":
                continue
            style = "-o" if meth == "adaptive" else "--s"
            lw = 2.6 if meth == "adaptive" else 1.5
            plt.plot(alts, series(meth, "tiles"), style, label=meth, linewidth=lw)
        plt.xlabel("Simulated altitude (m)")
        plt.ylabel("Tiles per frame (compute cost)")
        plt.title("Compute cost vs altitude: fixed slices are constant, adaptive scales")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "tiles_vs_altitude.png", dpi=130)
        plt.close()
        print(f"\nplots + metrics → {out}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
