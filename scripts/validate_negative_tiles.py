#!/usr/bin/env python3
"""Validate pre-existing gommapps negative tiles -> keep only clean, train-safe ones.

Each candidate tile must pass ALL checks to be kept as a training negative:
  1) parent frame is a LABELED gommapps frame assigned to the TRAIN split
     (drops val/test-parent tiles = leakage, and unreviewed-parent tiles = no GT);
  2) the tile rectangle does NOT overlap any GT box in its parent frame
     (drops tiles that clip a real whale = would poison training as "empty").

Tile geometry: the tile image's own (W,H) = tile size; the filename '..._x{X}_y{Y}...'
gives the tile's top-left origin in parent-frame pixels.

Kept tiles are symlinked (with empty labels) into --out; a manifest records every
candidate with its keep/drop reason.
"""
import argparse
import csv
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
POOLS = ["gommapps_raw", "gommapps"]
ORIGIN_RE = re.compile(r"_x(\d+)_y(\d+)")
PARENT_RE = re.compile(r"(NOAA_SEFSC_GoMMAPPS_[A-Za-z0-9]+_(?:Sight[0-9a-z]+|Misc|MISC)_IMG_[0-9]+)")
EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def load_train_stems(splits_csv):
    train = set()
    with splits_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["source"] == "gommapps" and row["split"] == "train":
                train.add(row["stem"])
    return train


def gt_boxes_px(stem, images_dir, labels_dir, dim_cache):
    """Return list of (x1,y1,x2,y2) px GT boxes for a parent frame, or None if no image."""
    img = next((images_dir / f"{stem}{e}" for e in EXTS
                if (images_dir / f"{stem}{e}").exists()), None)
    if img is None:
        return None
    if stem not in dim_cache:
        with Image.open(img) as im:
            dim_cache[stem] = im.size
    W, H = dim_cache[stem]
    lf = labels_dir / f"{stem}.txt"
    boxes = []
    if lf.exists():
        for ln in lf.read_text().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            boxes.append((( cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H))
    return boxes


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "splits" / "splits_v1.csv")
    ap.add_argument("--gommapps", type=Path, default=ROOT / "data" / "normalized" / "gommapps")
    ap.add_argument("--label-dir", default="labels_with_inferred")
    ap.add_argument("--hardneg-root", type=Path, default=ROOT / "data" / "hard_negatives")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "hard_negatives" / "gommapps_tiles_clean")
    args = ap.parse_args()

    train = load_train_stems(args.splits)
    images_dir = args.gommapps / "images"
    labels_dir = args.gommapps / args.label_dir
    dim_cache = {}

    out_img = args.out / "images"
    out_lbl = args.out / "labels"
    for d in (out_img, out_lbl):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    kept = 0
    reasons = {"kept": 0, "parent_not_train": 0, "overlaps_whale": 0, "no_geom": 0}
    seen = set()
    for pool in POOLS:
        pdir = args.hardneg_root / pool / "images"
        if not pdir.is_dir():
            continue
        for tile in sorted(pdir.glob("*.jpg")):
            pm = PARENT_RE.search(tile.name)
            om = ORIGIN_RE.search(tile.name)
            if not pm or not om:
                reasons["no_geom"] += 1
                rows.append([pool, tile.name, "", "drop:no_geom"])
                continue
            parent = pm.group(1)
            if parent not in train:
                reasons["parent_not_train"] += 1
                rows.append([pool, tile.name, parent, "drop:parent_not_train"])
                continue
            gts = gt_boxes_px(parent, images_dir, labels_dir, dim_cache)
            if gts is None:
                reasons["no_geom"] += 1
                rows.append([pool, tile.name, parent, "drop:parent_image_missing"])
                continue
            x, y = int(om.group(1)), int(om.group(2))
            with Image.open(tile) as im:
                tw, th = im.size
            trect = (x, y, x + tw, y + th)
            if any(overlaps(trect, g) for g in gts):
                reasons["overlaps_whale"] += 1
                rows.append([pool, tile.name, parent, "drop:overlaps_whale"])
                continue
            # dedup identical tile names across pools
            if tile.name in seen:
                rows.append([pool, tile.name, parent, "drop:duplicate"])
                continue
            seen.add(tile.name)
            link = out_img / tile.name
            if not link.exists():
                link.symlink_to(tile.resolve())
            (out_lbl / f"{tile.stem}.txt").write_text("")
            kept += 1
            reasons["kept"] += 1
            rows.append([pool, tile.name, parent, "keep"])

    with (args.out / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pool", "tile", "parent", "decision"])
        w.writerows(rows)

    print(f"clean negative tiles -> {args.out}")
    print(f"  candidates: {len(rows)}")
    for k, v in reasons.items():
        print(f"    {k}: {v}")
    print(f"  KEPT (train-safe, whale-free): {kept}")


if __name__ == "__main__":
    main()
