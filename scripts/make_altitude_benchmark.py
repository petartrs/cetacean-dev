#!/usr/bin/env python3
"""Build a controlled altitude-sweep benchmark for adaptive SAHI.

Why: no real dataset sweeps altitude, so we can't directly measure "recall vs altitude"
for fixed vs adaptive SAHI. This builds a *controlled semi-synthetic* benchmark from REAL
pixels — real animal crops on real empty-ocean tiles — where only the geometry is varied.

How (physically consistent):
  * Background = a big ocean canvas stitched from empty-water hard-negative tiles.
  * Animals   = crops of REAL, model-detectable GT animals (verified at native scale, so
                the benchmark isolates the effect of *slicing*, not intrinsic detectability).
  * For a simulated altitude → GSD (via a reference camera), an animal of real length L_m
    is placed at  animal_px = L_m / GSD  (its true apparent pixel size). Only DOWNSCALE is
    allowed (never upscale past the source crop) so no fake blur is introduced.
  * The SAME animals at the SAME positions are re-rendered across an altitude ladder →
    a clean sweep with exact YOLO ground truth at every level.

Outputs:
    <out>/frames/alt{A}m.jpg          synthetic survey frame per altitude
    <out>/labels/alt{A}m.txt          YOLO GT (class 0) per frame
    <out>/manifest.csv                frame, altitude_m, gsd_cm_px, n_animals, sizes
    <out>/master_preview.jpg          mid-altitude frame with GT boxes drawn

Anchor: conclusions are anchored by the one REAL APEM 151MP frame (see adaptive_sahi C1).

Usage:
    python scripts/make_altitude_benchmark.py \
        --out data/benchmark/altitude_sweep \
        --altitudes 60 90 120 160 200 260 320 \
        --canvas 8192 5460 --n-animals 14
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from cetacean.inference.adaptive_sahi import gsd_from_geometry, CAMERA_REGISTRY, DEFAULT_WEIGHTS

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Aerial (top-down) sources to draw REAL animal crops from. We need crops spanning a wide
# range of native resolution: big-in-frame whale sources (cetacean-detector, dryad) supply
# the large-bodied animals; the small-target sources (uav-porpoises, whale-bbox-layer)
# supply porpoise/dolphin-scale animals. Each placed animal is rendered at its assigned
# body length (physically consistent) and shrunk with simulated altitude.
SOURCES = ("cetacean-detector", "aerial-right-whale", "dryad", "whale-bbox-layer",
           "uav-porpoises")
NORM_ROOT = Path("data/normalized")


def sample_lengths(n: int, dist: str, rng: random.Random, *, fixed_len: float,
                   len_range: tuple, small_range: tuple, large_range: tuple,
                   large_frac: float):
    """Draw n body lengths (m) under the requested distribution.

    * ``fixed``     -- every animal is ``fixed_len`` m (legacy behaviour).
    * ``uniform``   -- U(len_range) : equal probability across the whole size range.
    * ``realistic`` -- two-component mixture reflecting NE-Atlantic aerial-survey
      assemblages (Hammond et al. 2013, Biol. Conserv. 164:107-122): small delphinids /
      harbour porpoise dominate (``small_range`` m) with occasional large baleen whales
      (``large_range`` m) at rate ``large_frac``. This also mirrors our own training set,
      whose apparent-size distribution is bimodal (porpoise mode vs whale mode).
    """
    if dist == "fixed":
        return [float(fixed_len)] * n
    if dist == "uniform":
        return [rng.uniform(*len_range) for _ in range(n)]
    if dist == "realistic":
        out = []
        for _ in range(n):
            if rng.random() < large_frac:
                out.append(rng.uniform(*large_range))
            else:
                out.append(rng.uniform(*small_range))
        return out
    raise ValueError(f"unknown --len-dist {dist}")


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


def collect_verified_animals(model, max_per_source: int, min_native: int,
                             iou_thr: float = 0.3):
    """Return [(patch_bgr, box_in_patch(x1,y1,x2,y2), native_long_px), ...].

    A GT box is kept only if (a) the model detects it at native scale (matched by IoU) and
    (b) its native longest side >= `min_native` px, so there is downscale headroom to shrink
    it across the whole altitude ladder without upscaling. Every kept animal is one the
    detector *can* find at native scale, isolating the effect of slicing.
    """
    kept = []
    for src in SOURCES:
        img_dir = NORM_ROOT / src / "images"
        lbl_dir = NORM_ROOT / src / "labels"
        if not img_dir.is_dir():
            print(f"  [skip] {src}: no images dir")
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        random.shuffle(imgs)
        n_src = 0
        for ip in imgs:
            if n_src >= max_per_source:
                break
            img = cv2.imread(str(ip))
            if img is None:
                continue
            H, W = img.shape[:2]
            gts = load_gt(lbl_dir / f"{ip.stem}.txt", W, H)
            if not gts:
                continue
            r = model.predict(source=str(ip), imgsz=1024, conf=0.25, verbose=False)[0]
            dets = r.boxes.xyxy.cpu().tolist() if (r.boxes is not None and len(r.boxes)) else []
            for g in gts:
                if n_src >= max_per_source:
                    break
                gx1, gy1, gx2, gy2 = (int(round(v)) for v in g)
                bw, bh = gx2 - gx1, gy2 - gy1
                if max(bw, bh) < min_native:
                    continue  # not enough native resolution for the sweep
                if not any(iou(d, g) >= iou_thr for d in dets):
                    continue  # not detected at native scale → skip
                mx, my = int(bw * 0.5), int(bh * 0.5)          # water margin for blending
                px1, py1 = max(0, gx1 - mx), max(0, gy1 - my)
                px2, py2 = min(W, gx2 + mx), min(H, gy2 + my)
                patch = img[py1:py2, px1:px2].copy()
                if patch.size == 0:
                    continue
                box_in_patch = (gx1 - px1, gy1 - py1, gx2 - px1, gy2 - py1)
                kept.append((patch, box_in_patch, max(bw, bh)))
                n_src += 1
        print(f"  {src}: kept {n_src} verified high-res animals")
    return kept


def build_ocean_canvas(tiles: list[Path], W: int, H: int, rng: random.Random):
    """Stitch a big ocean canvas from random empty-water tiles (with random flips)."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ts = 0
    sample = cv2.imread(str(tiles[0]))
    ts = sample.shape[0] if sample is not None else 2048
    for y in range(0, H, ts):
        for x in range(0, W, ts):
            t = cv2.imread(str(rng.choice(tiles)))
            if t is None:
                continue
            if rng.random() < 0.5:
                t = cv2.flip(t, 1)
            if rng.random() < 0.5:
                t = cv2.flip(t, 0)
            th, tw = t.shape[:2]
            y2, x2 = min(y + th, H), min(x + tw, W)
            canvas[y:y2, x:x2] = t[:y2 - y, :x2 - x]
    return canvas


def feather_paste(canvas, patch, cx, cy):
    """Alpha-feathered paste of `patch` centred at (cx, cy). Water-on-water → soft seams."""
    ph, pw = patch.shape[:2]
    x1, y1 = int(cx - pw / 2), int(cy - ph / 2)
    x2, y2 = x1 + pw, y1 + ph
    H, W = canvas.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return False
    # feather mask: 1 in the centre, fading to 0 over a border ~15% of the patch
    m = np.ones((ph, pw), dtype=np.float32)
    b = max(4, int(min(ph, pw) * 0.15))
    ramp = np.linspace(0, 1, b, dtype=np.float32)
    m[:b, :] *= ramp[:, None]
    m[-b:, :] *= ramp[::-1, None]
    m[:, :b] *= ramp[None, :]
    m[:, -b:] *= ramp[None, ::-1]
    m = cv2.GaussianBlur(m, (0, 0), b / 3.0)[..., None]
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (patch.astype(np.float32) * m + region * (1 - m)).astype(np.uint8)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmark/altitude_sweep")
    ap.add_argument("--altitudes", type=float, nargs="+",
                    default=[60, 90, 120, 160, 200, 260, 320])
    ap.add_argument("--camera", default="dji_zenmuse_p1",
                    help="reference camera mapping altitude→GSD")
    ap.add_argument("--canvas", type=int, nargs=2, default=[8192, 5460])
    ap.add_argument("--n-animals", type=int, default=14)
    ap.add_argument("--max-per-source", type=int, default=8)
    ap.add_argument("--animal-len", type=float, default=3.0,
                    help="reference body length (m) for --len-dist fixed")
    ap.add_argument("--len-dist", choices=("fixed", "uniform", "realistic"),
                    default="fixed",
                    help="per-animal body-length distribution (fixed=legacy single size)")
    ap.add_argument("--len-range", type=float, nargs=2, default=[1.0, 20.0],
                    help="[min max] body length (m) for --len-dist uniform")
    ap.add_argument("--small-range", type=float, nargs=2, default=[1.5, 4.0],
                    help="small-animal length range (m) for --len-dist realistic")
    ap.add_argument("--large-range", type=float, nargs=2, default=[10.0, 20.0],
                    help="large-whale length range (m) for --len-dist realistic")
    ap.add_argument("--large-frac", type=float, default=0.15,
                    help="fraction of large whales for --len-dist realistic")
    ap.add_argument("--min-native", type=int, default=160,
                    help="min native longest-side px of source crops (downscale headroom)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--min-apparent", type=float, default=6.0, help="floor px (skip if smaller)")
    ap.add_argument("--jitter", type=float, default=0.06,
                    help="grid-centre jitter as a fraction of canvas size (use a smaller "
                         "value for denser layouts to avoid overlaps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out).resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    cam = CAMERA_REGISTRY[args.camera]
    W, H = args.canvas

    # ocean tiles (empty-water hard negatives)
    tiles = sorted(p for d in ("apem", "gommapps", "gommapps_raw")
                   for p in (Path("data/hard_negatives") / d / "images").glob("*.jpg"))
    if not tiles:
        raise SystemExit("no hard-negative ocean tiles found")
    print(f"ocean tiles: {len(tiles)}")

    print("collecting verified-detectable animals...")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    pool = collect_verified_animals(model, args.max_per_source, args.min_native)
    if not pool:
        raise SystemExit("no verified high-res animals found; lower --min-native")
    if len(pool) < args.n_animals:
        print(f"  only {len(pool)} animals available; using all")

    # Per-animal body lengths, then feasibility-match to crops: an animal of length L can
    # only be rendered (downscale-only) if some crop has native longest side >= L/gsd_min
    # (its apparent size at the LOWEST altitude, where it is largest). Assign the biggest
    # lengths to the highest-resolution crops (greedy descending match).
    gsd_min = gsd_from_geometry(cam.pixel_pitch_um, min(args.altitudes),
                                cam.default_focal_mm)
    lengths = sample_lengths(
        args.n_animals, args.len_dist, rng, fixed_len=args.animal_len,
        len_range=tuple(args.len_range), small_range=tuple(args.small_range),
        large_range=tuple(args.large_range), large_frac=args.large_frac)
    lengths.sort(reverse=True)
    pool.sort(key=lambda x: x[2], reverse=True)   # native_long desc
    chosen = []
    used = [False] * len(pool)
    for L in lengths:
        req = L / gsd_min
        pick = next((k for k in range(len(pool))
                     if not used[k] and pool[k][2] >= req), None)
        if pick is None:
            print(f"  [drop] no crop with native>={req:.0f}px for L={L:.1f}m")
            continue
        used[pick] = True
        patch, box, native_long = pool[pick]
        chosen.append((patch, box, native_long, L))
    if not chosen:
        raise SystemExit("no animals could be placed; widen --min-native / sources")
    rng.shuffle(chosen)
    ls = [c[3] for c in chosen]
    print(f"placing {len(chosen)} animals, L={min(ls):.1f}-{max(ls):.1f} m "
          f"(dist={args.len_dist}, median {sorted(ls)[len(ls)//2]:.1f} m)")

    # fixed layout: grid of centres with jitter, spaced for the largest rendering
    canvas0 = build_ocean_canvas(tiles, W, H, rng)
    cols = int(np.ceil(np.sqrt(len(chosen) * W / H)))
    rows = int(np.ceil(len(chosen) / cols))
    centres = []
    for i in range(len(chosen)):
        r, c = divmod(i, cols)
        cx = (c + 0.5) / cols * W + rng.uniform(-args.jitter, args.jitter) * W
        cy = (r + 0.5) / rows * H + rng.uniform(-args.jitter, args.jitter) * H
        centres.append((cx, cy))

    manifest = []
    mid_idx = len(args.altitudes) // 2
    for ai, alt in enumerate(args.altitudes):
        gsd = gsd_from_geometry(cam.pixel_pitch_um, alt, cam.default_focal_mm)  # m/px
        canvas = canvas0.copy()
        labels = []
        sizes = []
        for (patch, box, native_long, length_m), (cx, cy) in zip(chosen, centres):
            target_px = length_m / gsd                      # true apparent size
            scale = min(target_px / native_long, 1.0)       # only downscale
            app = native_long * scale
            if app < args.min_apparent:
                continue
            sp = cv2.resize(patch, (max(1, int(patch.shape[1] * scale)),
                                    max(1, int(patch.shape[0] * scale))),
                            interpolation=cv2.INTER_AREA)
            bx1, by1, bx2, by2 = (v * scale for v in box)
            ph, pw = sp.shape[:2]
            ox, oy = int(cx - pw / 2), int(cy - ph / 2)
            if not feather_paste(canvas, sp, cx, cy):
                continue
            ax1, ay1, ax2, ay2 = ox + bx1, oy + by1, ox + bx2, oy + by2
            cxn = ((ax1 + ax2) / 2) / W
            cyn = ((ay1 + ay2) / 2) / H
            wn = (ax2 - ax1) / W
            hn = (ay2 - ay1) / H
            labels.append(f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}")
            sizes.append(round(app))
        stem = f"alt{int(alt)}m"
        cv2.imwrite(str(out / "frames" / f"{stem}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / f"{stem}.txt").write_text("\n".join(labels))
        manifest.append({
            "frame": f"frames/{stem}.jpg", "altitude_m": alt,
            "gsd_cm_px": round(gsd * 100, 3), "n_animals": len(labels),
            "min_px": min(sizes) if sizes else 0, "max_px": max(sizes) if sizes else 0,
            "len_min_m": round(min(ls), 2), "len_max_m": round(max(ls), 2),
            "len_median_m": round(sorted(ls)[len(ls) // 2], 2),
        })
        print(f"  alt {alt:>4.0f}m  GSD {gsd*100:5.2f}cm/px  animals {len(labels):>2}  "
              f"apparent px {min(sizes) if sizes else 0}-{max(sizes) if sizes else 0}")
        if ai == mid_idx:
            prev = canvas.copy()
            for lab in labels:
                _, cxn, cyn, wn, hn = map(float, lab.split())
                x1 = int((cxn - wn / 2) * W); y1 = int((cyn - hn / 2) * H)
                x2 = int((cxn + wn / 2) * W); y2 = int((cyn + hn / 2) * H)
                cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.imwrite(str(out / "master_preview.jpg"), prev,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

    with open(out / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        wr.writeheader()
        wr.writerows(manifest)
    print(f"\nbenchmark → {out}")


if __name__ == "__main__":
    main()
