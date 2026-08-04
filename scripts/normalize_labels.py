#!/usr/bin/env python3
"""Normalize the confirmed-safe cetacean datasets into a single-class YOLO format.

All boxes collapse to one class:  0 = cetacean

whale-321z9 and whale-jybhw were reviewed and DROPPED: their annotations mark a
single anatomical part per image (dorsal fin / body / fluke), not the whole
animal, which is unsuitable for a whole-animal detector.

  aerial:
    - aerial-right-whale   (Roboflow YOLO)
    - whale-bbox-layer     (Roboflow YOLO)
    - uav-porpoises        (per-scene YOLO txt; tracking/ ignored)
    - dryad                (VIA polygon JSON -> bbox)
    - whaleshape           (COCO JSON-lines -> bbox; seg/keypoints ignored)
  surface:
    - cetacean-detection   (Roboflow YOLO)
    - cetacean-detector    (Roboflow YOLO)
    - dolphin              (Roboflow YOLO)
    - dolphin2             (Roboflow YOLO)

Output layout (raw/ is never modified):

    data/normalized/<source>/images/<prefixed>.jpg   -> symlink to raw image
    data/normalized/<source>/labels/<prefixed>.txt   -> rewritten YOLO label (class 0)
    data/normalized/manifest.csv                      -> provenance for every image

Normalized filenames are prefixed with the source (and keep the scene id for
uav-porpoises) so they stay globally unique when later merged into one dataset.

Usage:
    python scripts/normalize_labels.py            # normalize all 8 sources
    python scripts/normalize_labels.py dryad      # normalize a single source
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUT_ROOT = PROJECT_ROOT / "data" / "normalized"

USE_SYMLINKS = True  # symlink images (save disk); labels are always real files

# Drop byte-identical images (keep first seen) across all sources. This removes
# exact duplicates that would otherwise leak across train/val/test -- e.g. the
# dryad images that appear in both train and test, and repeated uav-porpoises
# video frames.
DEDUP_BY_CONTENT = True

# Collapse Roboflow offline-augmentation groups: exports duplicate each source
# image as original + flipped + brightness-shifted copies (same filename before
# the ".rf.<hash>" tail). Keep one representative per original. Harmless for
# sources without augmentation (every base is already unique).
COLLAPSE_ROBOFLOW_AUG = True

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# populated at runtime, reset in main()
_SEEN_HASHES: dict[str, str] = {}   # md5 -> normalized image name already emitted
_USED_NAMES: dict[str, str] = {}    # normalized stem -> md5 (collision detection)

# dryad polygon -> bbox strategy:
#   "body"  : one box per region whose body_part == "body" (safe for multi-animal);
#             images that have regions but no "body" fall back to boxing every region.
#   "all"   : every region becomes its own box.
#   "union" : all regions in an image merge into a single box (best if 1 animal/image).
DRYAD_BOX_MODE = "body"

# source name -> (domain, format, raw relative path, split map)
# split map: raw sub-directory -> canonical split (train/val/test)
SOURCES: dict[str, dict] = {
    "aerial-right-whale": {
        "domain": "aerial",
        "format": "roboflow",
        "path": "aerial/labeled/aerial-right-whale",
        "splits": {"train": "train", "valid": "val", "test": "test"},
    },
    "whale-bbox-layer": {
        "domain": "aerial",
        "format": "roboflow",
        "path": "aerial/labeled/whale-bbox-layer",
        "splits": {"train": "train", "valid": "val", "test": "test"},
    },
    "uav-porpoises": {
        "domain": "aerial",
        "format": "uav_scenes",
        "path": "aerial/labeled/uav-porpoises",
        "splits": {"train": "train", "val": "val", "test": "test"},
    },
    "dryad": {
        "domain": "aerial",
        "format": "via",
        "path": "aerial/labeled/dryad",
        "splits": {
            "train_subset_1": "train",
            "train_subset_2": "train",
            "val": "val",
            "test": "test",
        },
    },
    "cetacean-detection": {
        "domain": "surface",
        "format": "roboflow",
        "path": "surface/labeled/cetacean-detection",
        "splits": {"train": "train", "valid": "val", "test": "test"},
    },
    "cetacean-detector": {
        "domain": "surface",
        "format": "roboflow",
        "path": "surface/labeled/cetacean-detector",
        "splits": {"train": "train", "valid": "val", "test": "test"},
    },
    "dolphin": {
        "domain": "surface",
        "format": "roboflow",
        "path": "surface/labeled/dolphin",
        "splits": {"train": "train", "valid": "val", "test": "test"},
    },
    "dolphin2": {
        "domain": "surface",
        "format": "roboflow",
        "path": "surface/labeled/dolphin2",
        "splits": {"train": "train", "valid": "val", "test": "test"},
    },
    "whaleshape": {
        "domain": "aerial",
        "format": "coco",
        "path": "aerial/labeled/dryad_doi_10_5061_dryad_6q573n668__v20260312",
        "images_subdir": "images",
        # COCO JSON-lines files -> canonical split
        "splits": {"train.json": "train", "test.json": "test"},
    },
}

CLASS_ID = 0  # cetacean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def link_image(src: Path, dst: Path) -> None:
    """Symlink (or copy) a raw image into the normalized tree, idempotently."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if USE_SYMLINKS:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def write_label(dst: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    """Write a YOLO label file (class id forced to CLASS_ID). Empty list => background."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        for (xc, yc, w, h) in boxes
    ]
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(
    img_path: Path,
    boxes: list,
    out_stem: str,
    source: str,
    domain: str,
    split: str,
    scene: str,
    out_img: Path,
    out_lbl: Path,
    rows: list,
    stats: dict,
    oriented_image=None,
) -> None:
    """Write one normalized image+label, with content dedup and collision-safe names.

    If ``oriented_image`` (a PIL image) is given, it is saved as a real file with
    EXIF orientation already baked into the pixels; otherwise the raw image is
    symlinked/copied as-is.
    """
    digest = md5_of(img_path)

    if DEDUP_BY_CONTENT and digest in _SEEN_HASHES:
        stats["dup_skipped"] = stats.get("dup_skipped", 0) + 1
        return

    # same stem but different content -> disambiguate so nothing is overwritten
    if out_stem in _USED_NAMES and _USED_NAMES[out_stem] != digest:
        out_stem = f"{out_stem}_{digest[:8]}"

    _SEEN_HASHES[digest] = out_stem
    _USED_NAMES[out_stem] = digest

    dst_img = out_img / f"{out_stem}{img_path.suffix}"
    if oriented_image is not None:
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        oriented_image.save(dst_img, quality=95)
    else:
        link_image(img_path, dst_img)
    write_label(out_lbl / f"{out_stem}.txt", boxes)

    stats["images"] += 1
    stats["boxes"] += len(boxes)
    if not boxes:
        stats["backgrounds"] += 1
    rows.append([source, domain, split, scene, f"{out_stem}{img_path.suffix}", len(boxes)])


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def remap_yolo_lines(text: str) -> list[tuple[float, float, float, float]]:
    """Parse YOLO label text and return boxes (coords kept, class dropped)."""
    boxes: list[tuple[float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        boxes.append((clamp01(xc), clamp01(yc), clamp01(w), clamp01(h)))
    return boxes


def roboflow_base(stem: str) -> str:
    """Original image name before Roboflow's '.rf.<hash>' augmentation tail."""
    return re.split(r"\.rf\.", stem, maxsplit=1)[0]


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

def convert_roboflow(name: str, cfg: dict, out_img: Path, out_lbl: Path, rows: list) -> dict:
    """Roboflow YOLO: iterate images (keeps backgrounds), remap all classes to 0."""
    stats = {"images": 0, "boxes": 0, "backgrounds": 0}
    src_root = RAW_ROOT / cfg["path"]
    for raw_split, split in cfg["splits"].items():
        images_dir = src_root / raw_split / "images"
        labels_dir = src_root / raw_split / "labels"
        if not images_dir.is_dir():
            continue
        imgs = sorted(p for p in images_dir.iterdir() if p.suffix in IMAGE_EXTS)

        if COLLAPSE_ROBOFLOW_AUG:
            groups: dict[str, list[Path]] = {}
            for img in imgs:
                groups.setdefault(roboflow_base(img.stem), []).append(img)
            kept: list[Path] = []
            for members in groups.values():
                members.sort()
                kept.append(members[0])  # one representative per original
                stats["aug_collapsed"] = stats.get("aug_collapsed", 0) + len(members) - 1
            imgs = sorted(kept)

        for img in imgs:
            label_file = labels_dir / f"{img.stem}.txt"
            boxes = remap_yolo_lines(label_file.read_text()) if label_file.exists() else []
            emit(img, boxes, f"{name}__{img.stem}", name, cfg["domain"], split, "",
                 out_img, out_lbl, rows, stats)
    return stats


def convert_uav_scenes(name: str, cfg: dict, out_img: Path, out_lbl: Path, rows: list) -> dict:
    """uav-porpoises: per-scene folders of frame.jpg + frame.txt (class already 0)."""
    stats = {"images": 0, "boxes": 0, "backgrounds": 0}
    src_root = RAW_ROOT / cfg["path"]
    for raw_split, split in cfg["splits"].items():
        split_dir = src_root / raw_split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            scene = scene_dir.name
            for img in sorted(scene_dir.glob("*.jpg")):
                label_file = img.with_suffix(".txt")
                boxes = remap_yolo_lines(label_file.read_text()) if label_file.exists() else []
                # stem already encodes scene, e.g. 001__0000
                emit(img, boxes, f"{name}__{img.stem}", name, cfg["domain"], split, scene,
                     out_img, out_lbl, rows, stats)
    return stats


def _via_regions(entry: dict) -> list[dict]:
    regions = entry.get("regions", [])
    if isinstance(regions, dict):
        return list(regions.values())
    return list(regions)


def _polygon_to_xyxy(shape: dict) -> tuple[float, float, float, float] | None:
    name = shape.get("name")
    if name in ("polygon", "polyline"):
        xs = shape.get("all_points_x") or []
        ys = shape.get("all_points_y") or []
        if not xs or not ys:
            return None
        return (min(xs), min(ys), max(xs), max(ys))
    if name == "rect":
        x, y = shape.get("x", 0), shape.get("y", 0)
        return (x, y, x + shape.get("width", 0), y + shape.get("height", 0))
    if name in ("circle", "point"):
        cx, cy = shape.get("cx", 0), shape.get("cy", 0)
        r = shape.get("r", 0)
        return (cx - r, cy - r, cx + r, cy + r)
    if name == "ellipse":
        cx, cy = shape.get("cx", 0), shape.get("cy", 0)
        rx, ry = shape.get("rx", 0), shape.get("ry", 0)
        return (cx - rx, cy - ry, cx + rx, cy + ry)
    return None


def _xyxy_to_yolo(box, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (clamp01(xc), clamp01(yc), clamp01(bw), clamp01(bh))


# EXIF orientation -> transform mapping a normalized point (x, y) in the raw
# stored-pixel frame to the corrected/display frame produced by
# ImageOps.exif_transpose. The 3/6/8 cases were confirmed empirically against the
# dryad images (rot180 / rot90CW / rot90CCW); 2/4/5/7 are the standard mirrored
# variants (absent in dryad but handled for completeness).
_ORIENT_XFORM = {
    1: lambda x, y: (x, y),
    2: lambda x, y: (1 - x, y),
    3: lambda x, y: (1 - x, 1 - y),
    4: lambda x, y: (x, 1 - y),
    5: lambda x, y: (y, x),
    6: lambda x, y: (1 - y, x),
    7: lambda x, y: (1 - y, 1 - x),
    8: lambda x, y: (y, 1 - x),
}


def _raw_bbox_to_yolo(box, rw: int, rh: int, orient: int) -> tuple[float, float, float, float]:
    """Normalize a raw-pixel bbox and rotate it into the EXIF-corrected frame."""
    x1, y1, x2, y2 = box
    xform = _ORIENT_XFORM.get(orient, _ORIENT_XFORM[1])
    corners = [
        xform(x1 / rw, y1 / rh), xform(x2 / rw, y2 / rh),
        xform(x1 / rw, y2 / rh), xform(x2 / rw, y1 / rh),
    ]
    nx1 = min(c[0] for c in corners); ny1 = min(c[1] for c in corners)
    nx2 = max(c[0] for c in corners); ny2 = max(c[1] for c in corners)
    xc = (nx1 + nx2) / 2; yc = (ny1 + ny2) / 2
    return (clamp01(xc), clamp01(yc), clamp01(nx2 - nx1), clamp01(ny2 - ny1))


def convert_via(name: str, cfg: dict, out_img: Path, out_lbl: Path, rows: list) -> dict:
    """dryad: VIA polygon JSON -> single-class bboxes. Needs Pillow for image sizes."""
    import json

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "dryad conversion needs Pillow. Install it with:  pip install Pillow"
        ) from exc

    stats = {"images": 0, "boxes": 0, "backgrounds": 0, "no_body_fallback": 0,
             "missing_img": 0, "exif_reoriented": 0}
    src_root = RAW_ROOT / cfg["path"]
    for raw_split, split in cfg["splits"].items():
        sub = src_root / raw_split
        via_json = sub / "via_region_data.json"
        if not via_json.is_file():
            continue
        data = json.loads(via_json.read_text())
        for entry in data.values():
            filename = entry.get("filename")
            if not filename:
                continue
            img_path = sub / filename
            if not img_path.exists():
                stats["missing_img"] += 1
                continue

            regions = _via_regions(entry)
            body = [r for r in regions if r.get("region_attributes", {}).get("body_part") == "body"]

            if DRYAD_BOX_MODE == "union" and regions:
                selected, mode_union = regions, True
            elif DRYAD_BOX_MODE == "all":
                selected, mode_union = regions, False
            else:  # "body"
                if body:
                    selected, mode_union = body, False
                else:
                    selected, mode_union = regions, False  # fallback: box every region
                    if regions:
                        stats["no_body_fallback"] += 1

            xyxy = [b for b in (_polygon_to_xyxy(r.get("shape_attributes", {})) for r in selected) if b]

            # VIA polygons are stored in the RAW pixel frame, but the raw pixels
            # are not rotated. Over half the dryad images carry an orientation flag
            # (3/6/8). We bake the rotation into the saved pixels (exif_transpose)
            # AND apply the matching rotation to the polygon coords, or the boxes
            # land 90/180 degrees off from the animal.
            oriented_image = None
            with Image.open(img_path) as im0:
                orient = (im0.getexif().get(274, 1) or 1)
                rw, rh = im0.size
                if orient != 1:
                    oriented_image = ImageOps.exif_transpose(im0).convert("RGB")
                    stats["exif_reoriented"] += 1

            if mode_union and xyxy:
                x1 = min(b[0] for b in xyxy)
                y1 = min(b[1] for b in xyxy)
                x2 = max(b[2] for b in xyxy)
                y2 = max(b[3] for b in xyxy)
                boxes = [_raw_bbox_to_yolo((x1, y1, x2, y2), rw, rh, orient)]
            else:
                boxes = [_raw_bbox_to_yolo(b, rw, rh, orient) for b in xyxy]

            emit(img_path, boxes, f"{name}__{img_path.stem}", name, cfg["domain"], split, "",
                 out_img, out_lbl, rows, stats, oriented_image=oriented_image)
    return stats


def convert_coco(name: str, cfg: dict, out_img: Path, out_lbl: Path, rows: list) -> dict:
    """COCO JSON-lines (WHALESHAPE) -> single-class bboxes. Needs Pillow for image sizes.

    Each line is one record: {image_id, file_name, annotations:[{bbox, bbox_mode, ...}]}.
    bbox_mode 1 = XYWH_ABS (x, y, w, h top-left, pixels); 0 = XYXY_ABS. Segmentation and
    keypoints are ignored (detection only). category_id is dropped (all -> class 0).
    """
    import json

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "COCO conversion needs Pillow. Install it with:  pip install Pillow"
        ) from exc

    stats = {"images": 0, "boxes": 0, "backgrounds": 0, "missing_img": 0}
    src_root = RAW_ROOT / cfg["path"]
    images_dir = src_root / cfg.get("images_subdir", "images")
    for json_name, split in cfg["splits"].items():
        jf = src_root / json_name
        if not jf.is_file():
            continue
        for line in jf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            filename = rec.get("file_name")
            if not filename:
                continue
            img_path = images_dir / filename
            if not img_path.exists():
                stats["missing_img"] += 1
                continue
            with Image.open(img_path) as im:
                w, h = im.size

            boxes: list[tuple[float, float, float, float]] = []
            for ann in rec.get("annotations", []):
                bb = ann.get("bbox")
                if not bb or len(bb) < 4:
                    continue
                if ann.get("bbox_mode", 1) == 0:      # XYXY_ABS
                    x1, y1, x2, y2 = bb[:4]
                    bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
                else:                                  # XYWH_ABS (1)
                    bx, by, bw, bh = bb[:4]
                xc = (bx + bw / 2) / w
                yc = (by + bh / 2) / h
                boxes.append((clamp01(xc), clamp01(yc), clamp01(bw / w), clamp01(bh / h)))

            emit(img_path, boxes, f"{name}__{img_path.stem}", name, cfg["domain"], split, "",
                 out_img, out_lbl, rows, stats)
    return stats


CONVERTERS = {
    "roboflow": convert_roboflow,
    "uav_scenes": convert_uav_scenes,
    "via": convert_via,
    "coco": convert_coco,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def normalize_source(name: str, cfg: dict, rows: list) -> dict:
    out_img = OUT_ROOT / name / "images"
    out_lbl = OUT_ROOT / name / "labels"
    # fresh start for this source
    for d in (out_img, out_lbl):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    converter = CONVERTERS[cfg["format"]]
    return converter(name, cfg, out_img, out_lbl, rows)


def main(argv: list[str]) -> int:
    selected = argv or list(SOURCES)
    unknown = [s for s in selected if s not in SOURCES]
    if unknown:
        print(f"Unknown source(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(SOURCES)}")
        return 2

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _SEEN_HASHES.clear()
    _USED_NAMES.clear()
    rows: list = []
    grand = {"images": 0, "boxes": 0, "backgrounds": 0, "dup_skipped": 0}

    print(f"Normalizing {len(selected)} source(s) -> {OUT_ROOT}\n")
    for name in selected:
        cfg = SOURCES[name]
        stats = normalize_source(name, cfg, rows)
        grand["images"] += stats["images"]
        grand["boxes"] += stats["boxes"]
        grand["backgrounds"] += stats.get("backgrounds", 0)
        grand["dup_skipped"] += stats.get("dup_skipped", 0)
        extra = " ".join(
            f"{k}={v}" for k, v in stats.items()
            if k not in ("images", "boxes", "backgrounds") and v
        )
        print(
            f"  {name:<20} [{cfg['domain']:<7}] "
            f"images={stats['images']:>6} boxes={stats['boxes']:>6} "
            f"bg={stats.get('backgrounds', 0):>5} {extra}"
        )

    manifest = OUT_ROOT / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "domain", "orig_split", "scene", "image", "n_boxes"])
        writer.writerows(rows)

    print(
        f"\nDone. images={grand['images']} boxes={grand['boxes']} "
        f"backgrounds={grand['backgrounds']} duplicates_skipped={grand['dup_skipped']}"
    )
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
