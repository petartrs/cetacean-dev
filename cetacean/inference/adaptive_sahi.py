#!/usr/bin/env python3
"""Adaptive SAHI — choose the SAHI slice size *per frame* from imaging geometry.

The whole idea: plain inference resizes the entire frame to `imgsz` (1024), so on a
huge survey photo tiny animals shrink below the detection floor. SAHI fixes this by
running the model on overlapping crops, but a *fixed* crop size is only right for one
altitude. This module computes the crop size from the ground-sample-distance (GSD) so
that whatever the altitude/resolution, an animal lands at the object size the model was
trained on (~`target_px`).

Design note — two coupled SAHI knobs:
  * S = slice (crop) size, in original-image pixels.
  * I = per-tile inference size (the detection model's `image_size`).
An object of native size `a` px is seen by the model as `a * I / S` px. We FIX
`I = imgsz = 1024` (the trained resolution) and vary only `S`, so:
      apparent = a * 1024 / S      →      S = a * 1024 / target_px
Adaptivity is therefore purely in the crop size; tiles are always inferred at the
trained scale.

GSD cascade (priority):
  1. USER override   — direct GSD, or forced slice, or camera intrinsics + altitude.
  2. AUTO            — EXIF/XMP (altitude + focal) for stills, telemetry for live video.
  3. BLIND SWEEP     — no geometry at all → try several slices, pick the most consistent.

The two public functions `choose_slice_size` and `run_adaptive` are import-safe so the
ROS2 node (Phase 4) reuses the exact same logic with a live altitude source.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

# The locked pipeline model (final project baseline B, YOLO11n). Scripts override via --weights.
DEFAULT_WEIGHTS = (
    "/home/dell/cetacean-detection-final/runs/train/B/weights/best.pt"
)

# Trained object-size sweet spot (Phase 0.5, aerial effective longer-side px @ imgsz1024:
# p25=26, median=45, porpoise p50=30). Aim animals at ~45 px apparent.
DEFAULT_TARGET_PX = 45.0
DEFAULT_TARGET_LEN_M = 3.0   # generic cetacean body length (m); override per species.
DEFAULT_IMGSZ = 1024


# --------------------------------------------------------------------------- cameras
@dataclass
class Camera:
    name: str
    pixel_pitch_um: float
    default_focal_mm: float | None = None
    sensor_w_mm: float | None = None
    sensor_h_mm: float | None = None


# Pixel pitch derived from sensor width / pixel count where datasheets vary.
CAMERA_REGISTRY: dict[str, Camera] = {
    # 151 MP medium format (APEM survey camera). 53.4 mm / 14204 px ≈ 3.76 µm.
    "phaseone_ixm_rs150f": Camera(
        "Phase One iXM-RS150F", pixel_pitch_um=3.76, default_focal_mm=70.0,
        sensor_w_mm=53.4, sensor_h_mm=40.0),
    # 45 MP full frame. 35.9 mm / 8192 px ≈ 4.39 µm.
    "dji_zenmuse_p1": Camera(
        "DJI Zenmuse P1", pixel_pitch_um=4.39, default_focal_mm=35.0,
        sensor_w_mm=35.9, sensor_h_mm=24.0),
    # 61 MP full frame. 35.7 mm / 9504 px ≈ 3.76 µm.
    "sony_ilx_lr1": Camera(
        "Sony ILX-LR1", pixel_pitch_um=3.76, default_focal_mm=35.0,
        sensor_w_mm=35.7, sensor_h_mm=23.8),
    # 61 MP full frame (same sensor class as ILX-LR1). 35.7 mm / 9504 px ≈ 3.76 µm.
    "sony_a7r4": Camera(
        "Sony a7R IV", pixel_pitch_um=3.76, default_focal_mm=35.0,
        sensor_w_mm=35.7, sensor_h_mm=23.8),
}


def gsd_from_geometry(pixel_pitch_um: float, altitude_m: float, focal_mm: float) -> float:
    """Ground sample distance in metres/pixel:  GSD = pixel_pitch * altitude / focal."""
    return (pixel_pitch_um * 1e-6) * altitude_m / (focal_mm * 1e-3)


# --------------------------------------------------------------------------- decision
@dataclass
class SliceDecision:
    slice_size: int | None   # None = plain inference (skip SAHI); see `mode`
    mode: str                # "plain" | "sahi" | "blind_sweep"
    gsd: float | None
    animal_px: float | None
    apparent_px: float | None
    source: str              # where the GSD/decision came from
    reason: str


def choose_slice_size(
    img_w: int,
    img_h: int,
    *,
    gsd: float | None = None,
    camera: str | Camera | None = None,
    altitude_m: float | None = None,
    focal_mm: float | None = None,
    pixel_pitch_um: float | None = None,
    target_len_m: float = DEFAULT_TARGET_LEN_M,
    target_px: float = DEFAULT_TARGET_PX,
    skip_px: float | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    min_slice: int = 512,
    max_slice: int | None = None,
    force_slice: int | None = None,
    max_len_m: float | None = None,
    max_fill: float = 1.0,
) -> SliceDecision:
    """Decide the SAHI crop size for one frame. Pure arithmetic — safe to call per-frame.

    Returns a SliceDecision. `mode`:
      * "plain"       → run whole-frame inference (animal already big enough, or forced).
      * "sahi"        → run SAHI with `slice_size`.
      * "blind_sweep" → no geometry available; caller should run `blind_sweep_pick`.

    Size-range mode (`max_len_m` set): when the survey spans a range of body lengths,
    a single magnification target is wrong — a crop tuned to magnify small animals
    fragments the large ones across tile borders (precision collapse). Instead we size
    the crop so the *largest* expected animal (``max_len_m``) occupies at most ``max_fill``
    of a tile, guaranteeing it fits whole in some overlapping tile, while keeping the crop
    as small as the geometry allows so small animals are still magnified. The crop then
    shrinks smoothly with altitude, tracing the accuracy envelope.
    """
    long_side = max(img_w, img_h)
    if max_slice is None:
        max_slice = long_side  # can't crop bigger than the image

    # Tier 0: explicit slice override (user knows best).
    if force_slice is not None:
        fs = max(min_slice, min(int(force_slice), int(max_slice)))
        return SliceDecision(fs, "sahi", None, None, None, "user_slice",
                             f"forced slice={fs}")

    # Resolve GSD.
    source = None
    if gsd is not None:
        source = "user_gsd"
    elif altitude_m is not None:
        cam = CAMERA_REGISTRY.get(camera) if isinstance(camera, str) else camera
        pp = pixel_pitch_um or (cam.pixel_pitch_um if cam else None)
        fl = focal_mm or (cam.default_focal_mm if cam else None)
        if pp and fl:
            gsd = gsd_from_geometry(pp, altitude_m, fl)
            source = "camera+altitude"

    if gsd is None:
        return SliceDecision(None, "blind_sweep", None, None, None, "no_geometry",
                             "no GSD available → caller falls back to blind sweep")

    animal_px = target_len_m / gsd                 # animal size in native pixels
    downscale = long_side / imgsz                  # plain-inference resize factor
    apparent = animal_px / downscale               # px the plain model would see

    # Skip threshold = detection floor (trained apparent scale). Independent of the
    # magnification target used when we DO tile: skip if the animal is already
    # comfortably detectable at plain inference.
    skip = skip_px if skip_px is not None else target_px
    if apparent >= skip:
        return SliceDecision(None, "plain", gsd, animal_px, apparent, source,
                             f"animal ≈{apparent:.0f}px at plain ≥ skip {skip:.0f} "
                             f"→ skip SAHI")

    # Size-range mode: crop sized so the largest expected animal fits one tile.
    if max_len_m is not None:
        max_animal_px = max_len_m / gsd
        slice_size = int(round(max_animal_px / max(1e-6, max_fill)))
        slice_size = max(min_slice, min(slice_size, int(max_slice)))
        return SliceDecision(slice_size, "sahi", gsd, animal_px, apparent, source,
                             f"range: max animal {max_animal_px:.0f}px "
                             f"(fill {max_fill:.2f}) → slice {slice_size}")

    slice_size = int(round(animal_px * imgsz / target_px))
    slice_size = max(min_slice, min(slice_size, int(max_slice)))
    return SliceDecision(slice_size, "sahi", gsd, animal_px, apparent, source,
                         f"GSD {gsd*100:.2f} cm/px, animal {animal_px:.0f}px "
                         f"→ slice {slice_size}")


# --------------------------------------------------------------------------- EXIF/XMP
def read_exif_geometry(image_path: str | Path) -> dict:
    """Extract altitude / focal / camera from EXIF (+ DJI XMP RelativeAltitude fallback)."""
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS

    Image.MAX_IMAGE_PIXELS = None
    out = {"make": None, "model": None, "focal_mm": None, "altitude_m": None}
    try:
        im = Image.open(image_path)
        ex = im.getexif()
        tags = {TAGS.get(k, k): v for k, v in ex.items()}
        out["make"] = tags.get("Make")
        out["model"] = tags.get("Model")
        if tags.get("FocalLength") is not None:
            out["focal_mm"] = float(tags["FocalLength"])
        gps = ex.get_ifd(0x8825) if hasattr(ex, "get_ifd") else {}
        g = {GPSTAGS.get(k, k): v for k, v in gps.items()}
        if "GPSAltitude" in g:
            out["altitude_m"] = float(g["GPSAltitude"])
    except Exception:
        pass
    if out["altitude_m"] is None:
        rel = _read_xmp_altitude(image_path)
        if rel is not None:
            out["altitude_m"] = rel
    return out


def _read_xmp_altitude(image_path: str | Path, chunk: int = 512 * 1024) -> float | None:
    """DJI drones store above-ground altitude in XMP, not standard EXIF. Scan the header."""
    try:
        with open(image_path, "rb") as f:
            head = f.read(chunk)
    except Exception:
        return None
    for key in (b"drone-dji:RelativeAltitude", b"RelativeAltitude", b"GPSAltitude"):
        i = head.find(key)
        if i != -1:
            m = re.search(rb"[-+]?\d+\.?\d*", head[i + len(key):i + len(key) + 40])
            if m:
                try:
                    return abs(float(m.group()))
                except ValueError:
                    pass
    return None


# --------------------------------------------------------------------------- detection
def _build_sahi_model(weights: str, conf: float, imgsz: int, device: str = "cuda:0"):
    """SAHI detection model. Tiles are always inferred at `imgsz` (the trained scale)."""
    from sahi import AutoDetectionModel
    try:
        return AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=weights,
            confidence_threshold=conf, device=device, image_size=imgsz)
    except Exception:
        return AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=weights,
            confidence_threshold=conf, device=device, image_size=imgsz)


def run_adaptive(
    image_path: str | Path,
    weights: str,
    decision: SliceDecision,
    *,
    conf: float = 0.3,
    overlap: float = 0.2,
    imgsz: int = DEFAULT_IMGSZ,
    device: str = "cuda:0",
    yolo_model=None,
    sahi_model=None,
) -> list[tuple[float, float, float, float, float]]:
    """Run detection per the decision. Returns [(x1, y1, x2, y2, score), ...].

    Pass a preloaded `yolo_model` (plain) or `sahi_model` to avoid reloading weights in a
    per-frame loop (e.g. the ROS node / sweep eval). `image_path` may be a path (str/Path)
    or an in-memory BGR image (numpy array), so live ROS frames work without a temp file.
    """
    src = str(image_path) if isinstance(image_path, (str, Path)) else image_path
    if decision.mode == "plain" or decision.slice_size is None:
        from ultralytics import YOLO
        m = yolo_model or YOLO(weights)
        r = m.predict(source=src, imgsz=imgsz, conf=conf, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()):
                out.append((b[0], b[1], b[2], b[3], s))
        return out

    from sahi.predict import get_sliced_prediction
    model = sahi_model or _build_sahi_model(weights, conf, imgsz, device)
    # SAHI reads paths as RGB; an in-memory frame is BGR (cv2/ROS) and must be
    # converted or the model sees swapped channels and under-detects.
    sahi_src = src if isinstance(src, str) else cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    res = get_sliced_prediction(
        sahi_src, model,
        slice_height=decision.slice_size, slice_width=decision.slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0)
    out = []
    for op in res.object_prediction_list:
        x1, y1, x2, y2 = op.bbox.to_xyxy()
        out.append((x1, y1, x2, y2, op.score.value))
    return out


# --------------------------------------------------------------------------- ensemble
Detection = tuple  # (x1, y1, x2, y2, score)


def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def consensus_filter(dets_a, dets_b, iou_thr: float = 0.2):
    """Strict AND fusion: keep a detection from A only if some B detection corroborates it.

    Maximises precision — a proposer model (high recall) is gated by a confirmer model
    (high precision), so detections neither model agrees on (e.g. whitecap false positives
    only the porpoise-sensitive model fires on) are dropped.
    """
    return [d for d in dets_a if any(_iou_xyxy(d, e) >= iou_thr for e in dets_b)]


def weighted_box_fusion(det_lists, weights=None, iou_thr: float = 0.5,
                        conf_thr: float = 0.0):
    """Weighted Box Fusion (Solovyev et al.) of detections from several models.

    Unlike NMS/consensus (which keep or drop whole boxes), WBF *merges* overlapping boxes
    into one whose coordinates and score are confidence-weighted averages. A box confirmed
    by several models is boosted; a box seen by only one model is down-weighted toward that
    model's share of the total trust — so `weights` is the precision/recall dial (weight the
    high-recall model up when misses are costly, the high-precision model up in clutter).

    det_lists : list of per-model detection lists, each detection (x1, y1, x2, y2, score).
    weights   : per-model trust weights (default all 1.0).
    conf_thr  : drop fused boxes below this score (raise it to recover consensus-like precision).
    Returns fused [(x1, y1, x2, y2, score), ...] sorted by score.
    """
    n = len(det_lists)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0] * n
    w_total = float(sum(weights)) or 1.0

    entries = []  # (fused_weight = score*model_weight, model_idx, box)
    for mi, dl in enumerate(det_lists):
        wm = weights[mi]
        for d in dl:
            entries.append((d[4] * wm, mi, [d[0], d[1], d[2], d[3]]))
    entries.sort(key=lambda e: e[0], reverse=True)

    clusters = []  # each: {"coord_acc": [..], "wsum": float, "reps": current fused xyxy}
    for fw, mi, box in entries:
        best_j, best_iou = -1, iou_thr
        for j, c in enumerate(clusters):
            v = _iou_xyxy(box, c["rep"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j == -1:
            clusters.append({
                "coord_acc": [c * fw for c in box],
                "wsum": fw,
                "rep": list(box),
            })
        else:
            c = clusters[best_j]
            c["coord_acc"] = [a + b * fw for a, b in zip(c["coord_acc"], box)]
            c["wsum"] += fw
            c["rep"] = [a / c["wsum"] for a in c["coord_acc"]]

    out = []
    for c in clusters:
        score = min(1.0, c["wsum"] / w_total)
        if score >= conf_thr:
            x1, y1, x2, y2 = c["rep"]
            out.append((x1, y1, x2, y2, score))
    out.sort(key=lambda d: d[4], reverse=True)
    return out


def fuse_detections(det_lists, method: str = "wbf", *, weights=None,
                    iou_thr: float | None = None, conf_thr: float = 0.0):
    """Dispatch fusion of several models' detections.

    method="consensus" -> strict agreement (precision); expects exactly two lists
    [proposer, confirmer]. method="wbf" -> weighted box fusion (tunable via `weights`).
    """
    if method == "consensus":
        thr = 0.2 if iou_thr is None else iou_thr
        if len(det_lists) != 2:
            raise ValueError("consensus fusion expects [proposer, confirmer]")
        return consensus_filter(det_lists[0], det_lists[1], iou_thr=thr)
    if method == "wbf":
        thr = 0.5 if iou_thr is None else iou_thr
        return weighted_box_fusion(det_lists, weights=weights, iou_thr=thr, conf_thr=conf_thr)
    raise ValueError(f"unknown fusion method: {method}")


def ensemble_predict(image_path, model_specs, W, H, *, gsd=None, camera=None,
                    altitude_m=None, focal_mm=None, pixel_pitch_um=None,
                    fuse="wbf", weights=None, iou_thr=None, conf_thr=0.0,
                    overlap=0.2, imgsz=DEFAULT_IMGSZ, device="cuda:0"):
    """Run several models with per-model adaptive SAHI, then fuse.

    Each model spec is a dict {weights, target_px, target_len_m?, min_slice?, conf?} so every
    model tiles at *its own* scale anchor (slice = animal_px * imgsz / target_px), i.e.
    two-model adaptive SAHI. Returns (fused_dets, per_model) where per_model maps a label to
    its (decision, dets).
    """
    per_model = {}
    det_lists = []
    for spec in model_specs:
        dec = choose_slice_size(
            W, H, gsd=gsd, camera=camera, altitude_m=altitude_m, focal_mm=focal_mm,
            pixel_pitch_um=pixel_pitch_um,
            target_len_m=spec.get("target_len_m", DEFAULT_TARGET_LEN_M),
            target_px=spec.get("target_px", DEFAULT_TARGET_PX),
            min_slice=spec.get("min_slice", 512), imgsz=imgsz)
        dets = run_adaptive(image_path, spec["weights"], dec,
                            conf=spec.get("conf", 0.3), overlap=overlap,
                            imgsz=imgsz, device=device)
        label = spec.get("label", spec["weights"])
        per_model[label] = (dec, dets)
        det_lists.append(dets)
    fused = fuse_detections(det_lists, method=fuse, weights=weights,
                            iou_thr=iou_thr, conf_thr=conf_thr)
    return fused, per_model


def blind_sweep_pick(
    image_path: str | Path,
    weights: str,
    slices: list[int],
    *,
    conf: float = 0.3,
    overlap: float = 0.2,
    imgsz: int = DEFAULT_IMGSZ,
    device: str = "cuda:0",
    knee_ratio: float = 0.10,
) -> tuple[int, dict]:
    """Tier-4 fallback: no geometry. Run several crop sizes and pick the 'knee'.

    As the crop size shrinks (more, finer tiles) detections rise then plateau (and can
    start adding whitecap FPs). We pick the largest crop past which shrinking further adds
    < `knee_ratio` more detections — the cheapest slice that has already captured the scene.
    """
    from sahi.predict import get_sliced_prediction
    model = _build_sahi_model(weights, conf, imgsz, device)
    counts: dict[int, int] = {}
    src = str(image_path) if isinstance(image_path, (str, Path)) else image_path
    # SAHI expects RGB for ndarray input (see run_adaptive).
    sahi_src = src if isinstance(src, str) else cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    for sl in sorted(slices, reverse=True):          # large → small
        res = get_sliced_prediction(
            sahi_src, model,
            slice_height=sl, slice_width=sl,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0)
        counts[sl] = len(res.object_prediction_list)

    ordered = sorted(counts.items(), key=lambda kv: kv[0], reverse=True)  # by slice desc
    best = ordered[0][0]
    for i in range(1, len(ordered)):
        prev_c = ordered[i - 1][1]
        cur_c = ordered[i][1]
        gain = (cur_c - prev_c) / max(prev_c, 1)
        best = ordered[i][0]
        if gain < knee_ratio:
            best = ordered[i - 1][0]                 # last worthwhile (larger) slice
            break
    return best, counts


# --------------------------------------------------------------------------- viz
def draw(img, dets, color=(0, 0, 255)):
    for x1, y1, x2, y2, s in dets:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, f"{s:.2f}", (p1[0], max(0, p1[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Adaptive SAHI on a single image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    # geometry (user override tier)
    ap.add_argument("--camera", default=None, choices=sorted(CAMERA_REGISTRY) + [None])
    ap.add_argument("--altitude", type=float, default=None, help="altitude AGL in metres")
    ap.add_argument("--focal", type=float, default=None, help="focal length mm (override)")
    ap.add_argument("--pixel-pitch", type=float, default=None, help="sensor pixel pitch µm")
    ap.add_argument("--gsd", type=float, default=None, help="ground sample distance m/px (override)")
    ap.add_argument("--force-slice", type=int, default=None, help="force a fixed SAHI slice")
    ap.add_argument("--exif", action="store_true", help="read altitude/focal from EXIF/XMP")
    # target
    ap.add_argument("--target-len", type=float, default=DEFAULT_TARGET_LEN_M)
    ap.add_argument("--target-px", type=float, default=DEFAULT_TARGET_PX)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--min-slice", type=int, default=512)
    # inference
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--device", default="cuda:0")
    # blind sweep fallback
    ap.add_argument("--slices", type=int, nargs="+", default=[1024, 1536, 2048, 3072],
                    help="crop sizes used only when no geometry is available")
    # output
    ap.add_argument("--out", default=None, help="folder for annotated image + json")
    args = ap.parse_args()

    img_path = Path(args.image).resolve()
    if not img_path.exists():
        raise SystemExit(f"image not found: {img_path}")

    altitude, focal, camera = args.altitude, args.focal, args.camera
    if args.exif:
        exif = read_exif_geometry(img_path)
        print(f"EXIF: make={exif['make']} model={exif['model']} "
              f"focal={exif['focal_mm']} altitude={exif['altitude_m']}")
        altitude = altitude if altitude is not None else exif["altitude_m"]
        focal = focal if focal is not None else exif["focal_mm"]

    import cv2 as _cv2
    _cv2.setNumThreads(0)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(img_path) as im:
        W, H = im.size
    print(f"image: {img_path.name}  {W}x{H}  ({W*H/1e6:.1f} MP)")

    decision = choose_slice_size(
        W, H, gsd=args.gsd, camera=camera, altitude_m=altitude, focal_mm=focal,
        pixel_pitch_um=args.pixel_pitch, target_len_m=args.target_len,
        target_px=args.target_px, imgsz=args.imgsz, min_slice=args.min_slice,
        force_slice=args.force_slice)

    if decision.mode == "blind_sweep":
        best, counts = blind_sweep_pick(
            img_path, args.weights, args.slices, conf=args.conf,
            overlap=args.overlap, imgsz=args.imgsz, device=args.device)
        print(f"blind sweep counts: {counts} → picked slice {best}")
        decision = SliceDecision(best, "sahi", None, None, None, "blind_sweep",
                                 f"blind sweep knee → slice {best}")

    print(f"DECISION: mode={decision.mode} slice={decision.slice_size} "
          f"source={decision.source}\n  {decision.reason}")

    dets = run_adaptive(img_path, args.weights, decision, conf=args.conf,
                        overlap=args.overlap, imgsz=args.imgsz, device=args.device)
    scores = [d[4] for d in dets]
    smin = min(scores) if scores else 0.0
    smax = max(scores) if scores else 0.0
    savg = sum(scores) / len(scores) if scores else 0.0
    print(f"detections: {len(dets)}   score min/avg/max = {smin:.2f}/{savg:.2f}/{smax:.2f}")

    if args.out:
        out_dir = Path(args.out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        base = _cv2.imread(str(img_path))
        if base is not None:
            _cv2.imwrite(str(out_dir / f"{img_path.stem}_adaptive.jpg"), draw(base, dets))
        (out_dir / f"{img_path.stem}_adaptive.json").write_text(json.dumps(
            {"image": str(img_path), "size": [W, H], "decision": asdict(decision),
             "detections": [[float(x) for x in d] for d in dets]}, indent=2))
        print(f"saved → {out_dir}")


if __name__ == "__main__":
    main()
