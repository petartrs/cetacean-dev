#!/usr/bin/env python3
"""Assign every normalized image to a train/val/test split (staging manifest only).

Writes data/splits/splits_v1.csv (source, stem, split, role, group, boxes, kept).
No files are moved — Phase 4 build consumes this manifest. Leakage-safe grouping:

  gommapps          group = season+Sight### (atomic burst); Misc = singleton frames.
                    Sparse groups (few cetaceans/frame) biased into TEST; dense frames
                    kept in TRAIN. Uses labels_with_inferred. 70/15/15 by frame budget.
  uav-porpoises     SCENE-level 70/15/15 (whole scene one side), THEN every 5th frame
                    per scene kept (near-dup subsample). Dropped frames kept='0'.
  whale-bbox-layer  role=precision_test: ENTIRE source held out; every 5th frame per
                    Drone-Baleia scene kept. (High-GSD large-whale mAP test.)
  <other aerial>    seeded frame-level 70/15/15.

MAMMALS is a separate screening test set (not normalized here) and is excluded.
"""
import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORM = ROOT / "data" / "normalized"
SEED = 1337
R_TRAIN, R_VAL = 0.80, 0.15  # test = remainder (0.05)
R_TEST = round(1 - R_TRAIN - R_VAL, 4)

EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")

# per-source strategy
CFG = {
    "gommapps":           dict(label="labels_with_inferred", strat="gommapps"),
    "uav-porpoises":      dict(label="labels", strat="scene_stride", stride=5),
    "whale-bbox-layer":   dict(label="labels", strat="all_test_stride", stride=5,
                               role="precision_test"),
    "aerial-right-whale": dict(label="labels", strat="random"),
    "cetacean-detection": dict(label="labels", strat="random"),
    "cetacean-detector":  dict(label="labels", strat="random"),
    "dolphin":            dict(label="labels", strat="random"),
    "dolphin2":           dict(label="labels", strat="random"),
    "dryad":              dict(label="labels", strat="random"),
    "whaleshape":         dict(label="labels", strat="random"),
}

GOM_RE = re.compile(r"_(TO\d+\w+?)_(Sight\w+|Misc|MISC)_IMG_(\d+)", re.I)
UAV_RE = re.compile(r"__(\d+)__(\d+)$")
WBL_RE = re.compile(r"__(Drone-Baleia-\d+)_frame_(\d+)_")


def find_image(images_dir, stem):
    for e in EXTS:
        p = images_dir / f"{stem}{e}"
        if p.exists():
            return p
    return None


def box_count(lbl):
    if not lbl.exists():
        return 0
    return sum(1 for ln in lbl.read_text().splitlines() if ln.strip())


def frames_and_boxes(src, cfg):
    """Return list of (stem, boxes) for a source using its label dir."""
    images_dir = NORM / src / "images"
    labels_dir = NORM / src / cfg["label"]
    if not labels_dir.is_dir():
        labels_dir = NORM / src / "labels"
    out = []
    for lf in sorted(labels_dir.glob("*.txt")):
        if find_image(images_dir, lf.stem) is None:
            continue
        out.append((lf.stem, box_count(lf)))
    return out


def split_random(frames, rng):
    stems = [s for s, _ in frames]
    rng.shuffle(stems)
    n = len(stems)
    n_tr = int(round(n * R_TRAIN))
    n_va = int(round(n * R_VAL))
    assign = {}
    for i, s in enumerate(stems):
        assign[s] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assign


def split_scene_stride(frames, rng, stride):
    """uav: scene-level 70/15/15 by frame budget, then keep every `stride`-th frame."""
    scenes = defaultdict(list)
    for s, _ in frames:
        m = UAV_RE.search(s)
        scenes[m.group(1)].append((int(m.group(2)), s))
    scene_ids = list(scenes)
    rng.shuffle(scene_ids)
    total = len(frames)
    tr_cap, va_cap = total * R_TRAIN, total * (R_TRAIN + R_VAL)
    assign, kept = {}, {}
    run = 0
    for sc in scene_ids:
        split = "train" if run < tr_cap else ("val" if run < va_cap else "test")
        run += len(scenes[sc])
        for i, (fnum, stem) in enumerate(sorted(scenes[sc])):
            assign[stem] = split
            kept[stem] = 1 if (i % stride == 0) else 0
    return assign, kept


def split_all_test_stride(frames, stride):
    """whale-bbox-layer: everything -> test; keep every `stride`-th frame per scene."""
    scenes = defaultdict(list)
    for s, _ in frames:
        m = WBL_RE.search(s)
        scenes[m.group(1)].append((int(m.group(2)), s))
    assign, kept = {}, {}
    for sc, items in scenes.items():
        for i, (fnum, stem) in enumerate(sorted(items)):
            assign[stem] = "test"
            kept[stem] = 1 if (i % stride == 0) else 0
    return assign, kept


def split_gommapps(frames):
    """Empty frames -> train (hard negatives). Non-empty groups biased low-but-nonzero
    density into test, then val; densest remainder -> train. 80/15/5 budget."""
    groups = defaultdict(list)   # group -> [(stem, boxes)]
    for stem, b in frames:
        m = GOM_RE.search(stem)
        season, sight = m.group(1), m.group(2)
        gid = stem if sight.lower() == "misc" else f"{season}_{sight}"
        groups[gid].append((stem, b))
    tot_boxes = {g: sum(b for _, b in items) for g, items in groups.items()}
    dens = {g: tot_boxes[g] / len(items) for g, items in groups.items()}
    total = len(frames)
    te_cap, va_cap = total * R_TEST, total * R_VAL
    # empty-of-boxes groups -> train as hard negatives (never test/val)
    empty = [g for g in groups if tot_boxes[g] == 0]
    nonempty = [g for g in groups if tot_boxes[g] > 0]
    # low-but-nonzero density first: fill test then val, densest remainder -> train
    order = sorted(nonempty, key=lambda g: (dens[g], -len(groups[g])))
    assign = {}
    for g in empty:
        for stem, _ in groups[g]:
            assign[stem] = "train"
    te = va = 0
    for g in order:
        n = len(groups[g])
        if te < te_cap:
            split, te = "test", te + n
        elif va < va_cap:
            split, va = "val", va + n
        else:
            split = "train"
        for stem, _ in groups[g]:
            assign[stem] = split
    return assign, groups, dens


def group_of(src, stem):
    if src == "gommapps":
        m = GOM_RE.search(stem)
        return stem if m.group(2).lower() == "misc" else f"{m.group(1)}_{m.group(2)}"
    if src == "uav-porpoises":
        return UAV_RE.search(stem).group(1)
    if src == "whale-bbox-layer":
        return WBL_RE.search(stem).group(1)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "splits" / "splits_v1.csv")
    args = ap.parse_args()
    rng = random.Random(SEED)

    rows = []
    summary = defaultdict(lambda: defaultdict(int))     # source -> split -> kept frames
    box_sum = defaultdict(lambda: defaultdict(int))     # source -> split -> boxes (kept)
    dropped = defaultdict(int)

    for src, cfg in CFG.items():
        if not (NORM / src).is_dir():
            print(f"skip missing source: {src}")
            continue
        frames = frames_and_boxes(src, cfg)
        role = cfg.get("role", "train_pool")
        strat = cfg["strat"]
        kept = {s: 1 for s, _ in frames}
        if strat == "random":
            assign = split_random(frames, rng)
        elif strat == "scene_stride":
            assign, kept = split_scene_stride(frames, rng, cfg["stride"])
        elif strat == "all_test_stride":
            assign, kept = split_all_test_stride(frames, cfg["stride"])
        elif strat == "gommapps":
            assign, _, _ = split_gommapps(frames)
        boxes = dict(frames)
        for stem, _ in frames:
            k = kept.get(stem, 1)
            sp = assign[stem]
            rows.append([src, stem, sp, role, group_of(src, stem), boxes[stem], k])
            if k:
                summary[src][sp] += 1
                box_sum[src][sp] += boxes[stem]
            else:
                dropped[src] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "stem", "split", "role", "group", "boxes", "kept"])
        w.writerows(rows)

    print(f"\nsplit manifest -> {args.out}\n")
    print(f"{'source':<20}{'role':<15}{'train':>14}{'val':>14}{'test':>14}{'dropped':>9}")
    print("-" * 96)
    tot = defaultdict(int)
    for src, cfg in CFG.items():
        if src not in summary and dropped.get(src, 0) == 0:
            continue
        role = cfg.get("role", "train_pool")

        def cell(sp):
            f = summary[src].get(sp, 0)
            b = box_sum[src].get(sp, 0)
            tot[sp] += f
            return f"{f}f/{b}b"
        print(f"{src:<20}{role:<15}{cell('train'):>14}{cell('val'):>14}"
              f"{cell('test'):>14}{dropped.get(src, 0):>9}")
    print("-" * 96)
    print(f"{'TOTAL(kept frames)':<35}{tot['train']:>14}{tot['val']:>14}{tot['test']:>14}")


if __name__ == "__main__":
    main()
