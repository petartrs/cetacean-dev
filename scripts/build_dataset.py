#!/usr/bin/env python3
"""Materialise YOLO dataset variants for the composition experiments.

Consumes data/splits/splits_v1.csv + the negative pool (scripts/tile_negatives.py) and
emits symlinked YOLO datasets. Controlled design: val and test sets are FIXED (built once,
shared by every variant); only the TRAIN composition changes between variants, so metric
deltas are attributable to the manipulated factor.

Shared (built once, aerial only = drone-deployment target):
  data/dataset/shared/val            aerial val slices
  data/dataset/shared/test_precision whale-bbox-layer (held-out source, precision mAP)
  data/dataset/shared/test_aerial    aerial in-distribution 5% test slices

Per variant:
  data/dataset/<variant>/train       positives + dosed negatives
  data/dataset/<variant>/data.yaml   train=variant, val/test=shared

Negatives are drawn only from the gommapps pool (whale-free, all TRAIN-derived); dose is a
fraction of gommapps positive-train frames (284). B/E1/E2 = 20%, E4 sweep = 0/20/40%.
Single class 0 = cetacean. gommapps uses labels_with_inferred (realistic complete labels).
"""
import argparse
import csv
import hashlib
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORM = ROOT / "data" / "normalized"
POOL = ROOT / "data" / "hard_negatives" / "pool"
SEED = 1337
EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")

AERIAL_TRAIN = ["aerial-right-whale", "dryad", "whaleshape", "uav-porpoises", "gommapps"]
SURFACE_TRAIN = ["cetacean-detection", "cetacean-detector", "dolphin", "dolphin2"]
PRECISION_TEST_SRC = "whale-bbox-layer"

# dose = target fraction of the FINAL train set that is hard negatives (neg/(neg+pos))
VARIANTS = {
    "B":     dict(sources=AERIAL_TRAIN, dose=0.10),
    "E1":    dict(sources=AERIAL_TRAIN + SURFACE_TRAIN, dose=0.10),
    "E2":    dict(sources=[s for s in AERIAL_TRAIN if s != "uav-porpoises"], dose=0.10),
    "E4_0":  dict(sources=AERIAL_TRAIN, dose=0.00),
    "E4_20": dict(sources=AERIAL_TRAIN, dose=0.20),
}


def label_dir(source):
    return "labels_with_inferred" if source == "gommapps" else "labels"


def find_image(source, stem):
    d = NORM / source / "images"
    for e in EXTS:
        p = d / f"{stem}{e}"
        if p.exists():
            return p
    return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_splits(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            r["boxes"] = int(r["boxes"])
            r["kept"] = int(r["kept"])
            rows.append(r)
    return rows


def link_pair(img_dir, lbl_dir, source, stem, positive_only_empty=False):
    """Symlink normalized image + its label into a dataset dir. Returns True if written."""
    img = find_image(source, stem)
    if img is None:
        return False
    lbl = NORM / source / label_dir(source) / f"{stem}.txt"
    name = f"{source}__{stem}"
    dst_img = img_dir / f"{name}{img.suffix}"
    dst_lbl = lbl_dir / f"{name}.txt"
    if not dst_img.exists():
        dst_img.symlink_to(img.resolve())
    if lbl.exists():
        if not dst_lbl.exists():
            dst_lbl.symlink_to(lbl.resolve())
    else:
        dst_lbl.write_text("")
    return True


def add_negatives(img_dir, lbl_dir, pool_imgs, n):
    for p in pool_imgs[:n]:
        dst_img = img_dir / f"neg__{p.name}"
        dst_lbl = lbl_dir / f"neg__{p.stem}.txt"
        if not dst_img.exists():
            dst_img.symlink_to(p.resolve())
        dst_lbl.write_text("")


def mkdirs(base):
    img = base / "images"
    lbl = base / "labels"
    img.mkdir(parents=True, exist_ok=True)
    lbl.mkdir(parents=True, exist_ok=True)
    return img, lbl


def build_shared(rows, out):
    """val + test_precision + test_aerial, built once. Returns counts + held-out md5 set."""
    val_i, val_l = mkdirs(out / "shared" / "val")
    tp_i, tp_l = mkdirs(out / "shared" / "test_precision")
    ta_i, ta_l = mkdirs(out / "shared" / "test_aerial")
    n = dict(val=0, test_precision=0, test_aerial=0)
    held = set()
    for r in rows:
        if r["kept"] != 1:
            continue
        src, stem, split = r["source"], r["stem"], r["split"]
        img = find_image(src, stem)
        if src == PRECISION_TEST_SRC:
            if split == "test" and link_pair(tp_i, tp_l, src, stem):
                n["test_precision"] += 1
                held.add(md5(img))
            continue
        if src not in AERIAL_TRAIN:
            continue
        if split == "val" and link_pair(val_i, val_l, src, stem):
            n["val"] += 1
            held.add(md5(img))
        elif split == "test" and link_pair(ta_i, ta_l, src, stem):
            n["test_aerial"] += 1
            held.add(md5(img))
    return n, held


def build_variant(name, cfg, rows, pool_imgs, gom_pos, held, out):
    tr_i, tr_l = mkdirs(out / name / "train")
    sources = set(cfg["sources"])
    n_pos = 0
    n_leak = 0
    for r in rows:
        if r["kept"] != 1 or r["split"] != "train":
            continue
        src, stem = r["source"], r["stem"]
        if src not in sources:
            continue
        # gommapps empties come via the pool -> only positives here
        if src == "gommapps" and r["boxes"] == 0:
            continue
        # drop images identical to any held-out val/test frame (dryad/whaleshape overlap)
        img = find_image(src, stem)
        if img is not None and md5(img) in held:
            n_leak += 1
            continue
        if link_pair(tr_i, tr_l, src, stem):
            n_pos += 1
    # dose = fraction of the FINAL train set: neg/(neg+pos)=d -> neg = pos*d/(1-d)
    d = cfg["dose"]
    n_neg = min(round(n_pos * d / (1 - d)), len(pool_imgs)) if d > 0 else 0
    add_negatives(tr_i, tr_l, pool_imgs, n_neg)

    data_yaml = out / name / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        f"train: {name}/train/images\n"
        f"val: shared/val/images\n"
        f"test: shared/test_precision/images\n"
        f"nc: 1\n"
        f"names: [cetacean]\n"
    )
    return n_pos, n_neg, n_leak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "splits" / "splits_v1.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "dataset")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    rows = load_splits(args.splits)
    gom_pos = sum(1 for r in rows if r["source"] == "gommapps"
                  and r["split"] == "train" and r["kept"] == 1 and r["boxes"] > 0)

    pool_imgs = sorted(p for p in (POOL / "images").iterdir() if p.suffix.lower() in
                       (".jpg", ".jpeg", ".png"))
    random.Random(SEED).shuffle(pool_imgs)

    print(f"gommapps positive-train frames: {gom_pos}  |  negative pool: {len(pool_imgs)}")
    shared, held = build_shared(rows, args.out)
    print(f"shared  val={shared['val']}  test_precision={shared['test_precision']}  "
          f"test_aerial={shared['test_aerial']}  (held-out md5={len(held)})")

    for name in args.variants:
        pos, neg, leak = build_variant(name, VARIANTS[name], rows, pool_imgs, gom_pos, held, args.out)
        note = f"  [dropped {leak} leak-dupes]" if leak else ""
        print(f"{name:6s} train: {pos} positives + {neg} negatives = {pos + neg}  "
              f"(sources={len(VARIANTS[name]['sources'])}, dose={VARIANTS[name]['dose']:.0%}){note}")


if __name__ == "__main__":
    main()
