#!/usr/bin/env python3
"""Build a consolidated gommapps NEGATIVE pool at multiple scales.

The 57 confirmed-empty gommapps frames are whole-frame whale-free, so ANY crop of them
is a guaranteed-clean negative. We cut them at several tile sizes (default 1024, 2048)
with edge-flush coverage, and also keep the full frames. The 80 previously-validated
clean tiles (from positive frames' whale-free regions) are symlinked in as-is.

All 57 source frames are in the gommapps TRAIN split (empties were forced to train),
so nothing here can leak into val/test.

Output pool (--out):
  images/       tile crops (real jpg) + symlinks (full frames, validated tiles)
  labels/       empty .txt per image (YOLO negative)
  manifest.csv  image, kind(full|tile1024|tile2048|validated), parent, w, h
"""
import argparse
import csv
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def empty_frames(gommapps, label_dir):
    images_dir = gommapps / "images"
    labels_dir = gommapps / label_dir
    out = []
    for lf in sorted(labels_dir.glob("*.txt")):
        if lf.stat().st_size > 0:
            continue
        img = next((images_dir / f"{lf.stem}{e}" for e in EXTS
                    if (images_dir / f"{lf.stem}{e}").exists()), None)
        if img:
            out.append(img)
    return out


def edge_flush_origins(total, tile):
    """Origins covering [0,total) with tiles of size `tile`, last flush to the edge."""
    if tile >= total:
        return [0]
    origins = list(range(0, total - tile + 1, tile))
    if origins[-1] != total - tile:
        origins.append(total - tile)
    return origins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gommapps", type=Path, default=ROOT / "data" / "normalized" / "gommapps")
    ap.add_argument("--label-dir", default="labels_with_inferred")
    ap.add_argument("--validated", type=Path,
                    default=ROOT / "data" / "hard_negatives" / "gommapps_tiles_clean")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "hard_negatives" / "pool")
    ap.add_argument("--tiles", type=int, nargs="*", default=[1024, 2048])
    args = ap.parse_args()

    out_img = args.out / "images"
    out_lbl = args.out / "labels"
    for d in (out_img, out_lbl):
        d.mkdir(parents=True, exist_ok=True)

    rows = []

    def add_label(stem):
        (out_lbl / f"{stem}.txt").write_text("")

    frames = empty_frames(args.gommapps, args.label_dir)
    print(f"confirmed-empty source frames: {len(frames)}")

    for img in frames:
        with Image.open(img) as im:
            im = im.convert("RGB")
            W, H = im.size
            # full frame (symlink to original)
            link = out_img / f"{img.stem}__full.jpg"
            if not link.exists():
                link.symlink_to(img.resolve())
            add_label(f"{img.stem}__full")
            rows.append([link.name, "full", img.stem, W, H])
            # multi-scale tiles (real crops)
            for t in args.tiles:
                for x in edge_flush_origins(W, t):
                    for y in edge_flush_origins(H, t):
                        tw, th = min(t, W), min(t, H)
                        crop = im.crop((x, y, x + tw, y + th))
                        name = f"{img.stem}__t{t}_x{x}_y{y}.jpg"
                        crop.save(out_img / name, quality=88)
                        add_label(Path(name).stem)
                        rows.append([name, f"tile{t}", img.stem, tw, th])

    # validated clean tiles from positive frames (symlink in)
    vimg = args.validated / "images"
    n_val = 0
    if vimg.is_dir():
        for tile in sorted(vimg.glob("*.jpg")):
            link = out_img / f"val__{tile.name}"
            if not link.exists():
                link.symlink_to(tile.resolve())
            add_label(link.stem)
            with Image.open(tile) as im:
                w, h = im.size
            rows.append([link.name, "validated", tile.name, w, h])
            n_val += 1

    with (args.out / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "kind", "parent", "w", "h"])
        w.writerows(rows)

    by_kind = {}
    for r in rows:
        by_kind[r[1]] = by_kind.get(r[1], 0) + 1
    print(f"negative pool -> {args.out}")
    print(f"  total negatives: {len(rows)}")
    for k in sorted(by_kind):
        print(f"    {k}: {by_kind[k]}")


if __name__ == "__main__":
    main()
