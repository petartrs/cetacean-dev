#!/usr/bin/env python3
"""Box-scale analysis over normalized sources -> percentiles + deployment-GSD figure.

For every normalized source (data/normalized/<src>/{images,labels}) this reads each
box's native sqrt(area) in pixels (header-only image reads) and reports per-source and
overall percentiles, then renders a horizontal box/strip figure of the sqrt-area
distribution per source with SAHI tile-size reference context.

This is the Phase-2b "final" scale calc: same methodology as 1a but with the freshly
labeled gommapps folded in (using its complete certain+inferred label set).
"""
import argparse
import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PCTS = [5, 25, 50, 75, 95]
EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def find_image(images_dir, stem):
    for ext in EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "normalized")
    ap.add_argument("--out-fig", type=Path, default=ROOT / "runs" / "analysis" / "box_scale_2b.png")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "runs" / "analysis" / "box_scale_2b.csv")
    ap.add_argument("--tile", type=int, default=640, help="SAHI tile size for reference lines.")
    ap.add_argument("--label-dir", nargs="*", default=["gommapps=labels_with_inferred"],
                    help="Per-source label-dir overrides as src=dirname (default labels).")
    args = ap.parse_args()

    override = dict(x.split("=", 1) for x in args.label_dir)

    geo = defaultdict(list)      # sqrt(area) px per source
    n_img = defaultdict(int)
    n_bg = defaultdict(int)
    n_box = defaultdict(int)

    sources = sorted(d.name for d in args.root.iterdir()
                     if d.is_dir() and (d / "images").is_dir())
    for src in sources:
        images_dir = args.root / src / "images"
        lbl_name = override.get(src, "labels")
        labels_dir = args.root / src / lbl_name
        if not labels_dir.is_dir():
            labels_dir = args.root / src / "labels"
        dim_cache = {}
        for lf in sorted(labels_dir.glob("*.txt")):
            img = find_image(images_dir, lf.stem)
            if img is None:
                continue
            n_img[src] += 1
            key = img.resolve()
            if key not in dim_cache:
                with Image.open(img) as im:
                    dim_cache[key] = im.size
            W, H = dim_cache[key]
            lines = [ln.split() for ln in lf.read_text().splitlines() if ln.strip()]
            if not lines:
                n_bg[src] += 1
                continue
            for p in lines:
                w, h = float(p[3]) * W, float(p[4]) * H
                geo[src].append(sqrt(max(w, 0) * max(h, 0)))
                n_box[src] += 1

    sources = [s for s in sources if geo[s]]
    all_geo = [v for s in sources for v in geo[s]]

    def pct(vals):
        a = np.asarray(vals, float)
        return {p: float(np.percentile(a, p)) for p in PCTS}

    print(f"\nBox-scale (native sqrt-area px)  tile={args.tile}")
    print(f"{'source':<20}{'imgs':>6}{'bg':>5}{'boxes':>8}   {'p5':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p95':>7}")
    print("-" * 80)
    rows = []
    for s in sources + ["ALL"]:
        vals = all_geo if s == "ALL" else geo[s]
        pr = pct(vals)
        imgs = sum(n_img.values()) if s == "ALL" else n_img[s]
        bg = sum(n_bg.values()) if s == "ALL" else n_bg[s]
        bx = sum(n_box.values()) if s == "ALL" else n_box[s]
        print(f"{s:<20}{imgs:>6}{bg:>5}{bx:>8}   "
              f"{pr[5]:>7.1f}{pr[25]:>7.1f}{pr[50]:>7.1f}{pr[75]:>7.1f}{pr[95]:>7.1f}")
        rows.append([s, imgs, bg, bx] + [f"{pr[p]:.1f}" for p in PCTS])

    ap_all = pct(all_geo)
    print("-" * 80)
    print(f"median obj = {ap_all[50]:.1f}px = {100*ap_all[50]/args.tile:.1f}% of tile {args.tile}; "
          f"p95 = {ap_all[95]:.1f}px = {100*ap_all[95]/args.tile:.1f}% of tile")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "images", "background", "boxes"] + [f"p{p}" for p in PCTS])
        w.writerows(rows)

    # ---- figure: per-source sqrt-area boxplots (log-x) ----
    order = sorted(sources, key=lambda s: np.median(geo[s]))
    data = [geo[s] for s in order]
    labels = [f"{s}\n(n={n_box[s]})" for s in order]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 2))
    ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False,
               patch_artist=True, boxprops=dict(facecolor="#4c9fd6", alpha=0.7),
               medianprops=dict(color="#d62728", linewidth=1.6))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("native object size  sqrt(w·h)  [px]  (log scale)")
    ax.set_title(f"Aerial cetacean box scale by source (tile={args.tile}px)")
    ax.axvline(args.tile, color="k", ls="--", lw=1, label=f"tile {args.tile}px")
    ax.axvline(ap_all[50], color="#d62728", ls=":", lw=1.2,
               label=f"overall median {ap_all[50]:.0f}px")
    ax.grid(axis="x", which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"\nfigure -> {args.out_fig}\ncsv    -> {args.out_csv}")


if __name__ == "__main__":
    main()
