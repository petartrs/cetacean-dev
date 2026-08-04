#!/usr/bin/env python3
"""Screening-regime evaluation harness: SAHI-tiled inference + FROC / recall@FPPI.

Why this exists (see plan.md): the MAMMALS test is the LOW-GSD *screening* regime.
Objects are a handful of pixels in 6480x2160 frames, so plain full-frame inference
misses everything (that is why plain val ~= 0). SAHI slicing is mandatory. And the
right metric here is NOT mAP but a screening/cueing metric: how many real animals do
we catch (recall) at a tolerable false-alarm rate (false-positives-per-image). The
FROC curve plots recall vs FPPI as the confidence threshold is swept; recall@FPPI
reads a single operating point off it.

Two stages, so re-scoring is cheap and reproducible:
  1) PREDICT  SAHI-slice each frame ONCE at a low conf floor; cache (score, box)
              lists + GT per image to JSON.
  2) SCORE    sweep confidence, greedy-match preds->GT (IoU or center-in-GT),
              build the FROC curve, read recall at target FPPI, write CSV + PNG.

Locked SAHI default: tile 640 / overlap 0.2 / imgsz 1024 (plan Phase 2a).

Example (full run, all 6 models):
  ./ml-venv/bin/python scripts/eval_screening_froc.py \
    --models B=runs/train/B/weights/best.pt E0_s=runs/train/E0_s/weights/best.pt \
             E1=runs/train/E1/weights/best.pt E2=runs/train/E2/weights/best.pt \
             E4_0=runs/train/E4_0/weights/best.pt E4_20=runs/train/E4_20/weights/best.pt \
    --images data/dataset/test_mammals/images \
    --labels data/dataset/test_mammals/labels

Dry run (validate pipeline on a few frames, one model):
  ... --models B=runs/train/B/weights/best.pt --limit 8
"""
import argparse
import csv
import json
import time
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# --------------------------------------------------------------------------- GT
def load_gt(label_path, W, H):
    """YOLO normalized cxcywh -> absolute xyxy pixel boxes."""
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


def center_in(pred, gt):
    """True if the prediction's center falls inside the GT box (scale-tolerant)."""
    cx, cy = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


# ------------------------------------------------------------------ prediction
def predict_model(model_path, images, lbl_dir, args):
    """SAHI-slice every image once at the conf floor; return per-image records."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    det = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(model_path),
        confidence_threshold=args.base_conf, image_size=args.imgsz,
        device=args.device,
    )
    records = []
    for i, img in enumerate(images, 1):
        res = get_sliced_prediction(
            str(img), det, slice_height=args.tile, slice_width=args.tile,
            overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
            verbose=0,
        )
        W, H = res.image_width, res.image_height
        gt = load_gt(lbl_dir / f"{img.stem}.txt", W, H)
        preds = [[float(op.score.value),
                  float(op.bbox.minx), float(op.bbox.miny),
                  float(op.bbox.maxx), float(op.bbox.maxy)]
                 for op in res.object_prediction_list]
        records.append({"name": img.name, "W": W, "H": H, "gt": gt, "preds": preds})
        if i % 20 == 0 or i == len(images):
            print(f"    [{i}/{len(images)}] sliced", flush=True)
    return records


def cache_path(out_dir, tag, args, n):
    return out_dir / (f"cache_{tag}_t{args.tile}_ov{args.overlap}"
                      f"_sz{args.imgsz}_cf{args.base_conf}_n{n}.json")


def get_records(model_path, tag, images, lbl_dir, args, out_dir):
    cp = cache_path(out_dir, tag, args, len(images))
    if cp.exists() and not args.rebuild:
        print(f"  cache hit: {cp.name}")
        return json.loads(cp.read_text())
    print(f"  predicting {tag} ({len(images)} frames) ...")
    records = predict_model(model_path, images, lbl_dir, args)
    cp.write_text(json.dumps(records))
    print(f"  cached -> {cp.name}")
    return records


# ----------------------------------------------------------------- scoring
def match_at(records, conf, mode, iou_thr):
    """Greedy score-desc matching at a confidence threshold. Returns counts."""
    TP = FP = FN = 0
    frames_hit = n_pos_frames = 0
    for rec in records:
        gt = rec["gt"]
        dets = sorted((p for p in rec["preds"] if p[0] >= conf), key=lambda x: -x[0])
        matched = set()
        for d in dets:
            box = d[1:5]
            best_j, best_score = -1, iou_thr if mode == "iou" else 0.0
            for j, g in enumerate(gt):
                if j in matched:
                    continue
                if mode == "iou":
                    ov = iou(box, g)
                    if ov >= best_score:
                        best_score, best_j = ov, j
                else:  # center-in-GT, tie-broken by smallest GT (closest fit)
                    if center_in(box, g):
                        best_j = j
                        break
            if best_j >= 0:
                matched.add(best_j)
                TP += 1
            else:
                FP += 1
        FN += len(gt) - len(matched)
        if gt:
            n_pos_frames += 1
            if matched:
                frames_hit += 1
    return TP, FP, FN, frames_hit, n_pos_frames


def froc_curve(records, args):
    """Sweep thresholds -> list of operating points (conf, recall, fppi, ...)."""
    total_gt = sum(len(r["gt"]) for r in records)
    n_img = len(records)
    scores = sorted({round(p[0], 4) for r in records for p in r["preds"]}, reverse=True)
    if len(scores) > args.n_thresh:  # subsample thresholds evenly to bound cost
        idx = [round(k * (len(scores) - 1) / (args.n_thresh - 1)) for k in range(args.n_thresh)]
        scores = [scores[k] for k in sorted(set(idx))]
    points = []
    for conf in scores:
        TP, FP, FN, fh, npf = match_at(records, conf, args.match, args.iou)
        recall = TP / total_gt if total_gt else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        fppi = FP / n_img if n_img else 0.0
        frame_recall = fh / npf if npf else 0.0
        points.append({"conf": conf, "tp": TP, "fp": FP, "fn": FN,
                       "recall": recall, "precision": precision, "fppi": fppi,
                       "frame_recall": frame_recall})
    return points, total_gt, n_img


def recall_at_fppi(points, target):
    """Best recall achievable without exceeding a false-alarm budget."""
    ok = [p for p in points if p["fppi"] <= target]
    if not ok:
        return None
    best = max(ok, key=lambda p: p["recall"])
    return best


# -------------------------------------------------------------------- plotting
def plot_froc(curves, targets, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for tag, pts in curves.items():
        xs = [p["fppi"] for p in pts]
        ys = [p["recall"] for p in pts]
        order = sorted(range(len(xs)), key=lambda k: xs[k])
        plt.plot([xs[k] for k in order], [ys[k] for k in order],
                 marker=".", ms=3, lw=1.2, label=tag)
    for t in targets:
        plt.axvline(t, color="grey", ls="--", lw=0.6)
    plt.xlabel("False positives per image (FPPI)")
    plt.ylabel("Recall (box-level)")
    plt.title("HiDef survey screening FROC (SAHI-tiled)")
    plt.ylim(0, 1)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    print(f"\nFROC plot -> {out_png}")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="tag=path/to/best.pt entries")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--neg-dir", default=None,
                    help="Optional dir of extra negative frames (no GT) to inject.")
    ap.add_argument("--out", default="runs/screening")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--base-conf", type=float, default=0.05,
                    help="Detector conf floor; FROC thresholds swept post-hoc.")
    ap.add_argument("--match", choices=["iou", "center"], default="iou")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--fppi", default="0.1,0.5,1.0,2.0")
    ap.add_argument("--n-thresh", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="Dry-run: cap #frames.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rebuild", action="store_true", help="Ignore cached preds.")
    args = ap.parse_args()

    root = Path("/home/dell/cetacean-detection-final")
    out_dir = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.neg_dir:
        negs = sorted(p for p in Path(args.neg_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        images += negs
        print(f"Injected {len(negs)} negative frames from {args.neg_dir}")
    if args.limit:
        images = images[: args.limit]
    lbl_dir = Path(args.labels)

    targets = [float(t) for t in args.fppi.split(",")]
    models = []
    for m in args.models:
        tag, path = m.split("=", 1)
        p = Path(path) if Path(path).is_absolute() else root / path
        if not p.exists():
            raise SystemExit(f"weights not found: {p}")
        models.append((tag, p))

    print(f"Frames: {len(images)} | SAHI tile={args.tile} ov={args.overlap} "
          f"imgsz={args.imgsz} | match={args.match} iou={args.iou}\n")

    curves = {}
    csv_rows = []
    summary = []
    timings = {}
    for tag, path in models:
        print(f"=== {tag} ===")
        t0 = time.perf_counter()
        records = get_records(path, tag, images, lbl_dir, args, out_dir)
        t_pred = time.perf_counter() - t0
        pts, total_gt, n_img = froc_curve(records, args)
        t_score = time.perf_counter() - t0 - t_pred
        timings[tag] = (t_pred, t_score)
        print(f"  [{tag}] predict {t_pred:.1f}s ({t_pred / max(1, len(images)):.3f} s/frame) "
              f"score {t_score:.1f}s")
        curves[tag] = pts
        for p in pts:
            csv_rows.append({"model": tag, **p})
        row = {"model": tag, "gt": total_gt, "frames": n_img}
        for t in targets:
            best = recall_at_fppi(pts, t)
            row[f"R@{t}"] = round(best["recall"], 3) if best else None
            row[f"conf@{t}"] = round(best["conf"], 3) if best else None
        summary.append(row)

    csv_path = out_dir / "froc_points.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "conf", "tp", "fp", "fn",
                                          "recall", "precision", "fppi", "frame_recall"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nFROC points -> {csv_path}")

    # summary table
    hdr = ["model", "gt", "frames"] + sum(
        ([f"R@{t}", f"conf@{t}"] for t in targets), [])
    print("\n=== recall@FPPI summary ===")
    print(" | ".join(f"{h:>8}" for h in hdr))
    print("-" * (11 * len(hdr)))
    for row in summary:
        print(" | ".join(f"{str(row.get(h, '')):>8}" for h in hdr))

    print("\n=== timing (SAHI predict) ===")
    tot = 0.0
    for tag, (tp, ts) in timings.items():
        tot += tp
        print(f"  {tag:>8}: predict {tp:7.1f}s ({tp / max(1, len(images)):.3f} s/frame)  score {ts:5.1f}s")
    print(f"  TOTAL predict {tot:.1f}s | {len(images)} frames x {len(models)} models")
    (out_dir / "timings.csv").write_text(
        "model,predict_s,score_s,frames\n"
        + "".join(f"{tag},{tp:.2f},{ts:.2f},{len(images)}\n" for tag, (tp, ts) in timings.items()))

    plot_froc(curves, targets, out_dir / "froc.png")


if __name__ == "__main__":
    main()
