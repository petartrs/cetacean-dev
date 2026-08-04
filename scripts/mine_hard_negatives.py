#!/usr/bin/env python3
"""Assemble a FULL-SIZE gommapps hard-negative pool from confirmed-empty frames.

Negatives are the gommapps frames the human reviewed and left with NO box (guaranteed
no boxable cetacean). They are kept FULL-SIZE (not tiled) to match how gommapps
positives are trained (close/large animals, plain full-image predict).

"Hardness" = how much the current detector false-fires on the frame. We run the
pre-label model at low confidence; frames where it produces (false) detections on
glint / wakes / whitecaps are the valuable hard negatives. Output includes a review
folder with those false detections drawn, named hardest-first.

Output (under --out):
  images/            symlinks to the full-size negative frames
  labels/            empty .txt (YOLO negative)
  review/            rankNNN_<nfp>fp_<stem>.jpg  (false dets drawn, downscaled)
  manifest.csv       stem, season, n_fp, max_conf, rank
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "normalized" / "gommapps")
    ap.add_argument("--label-dir", default="labels_with_inferred")
    ap.add_argument("--model", type=Path,
                    default=ROOT.parent / "cetacean-aerial-detection" / "models" / "cetacean_yolo11n.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "hard_negatives" / "gommapps_fullframe")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--review-max", type=int, default=1600, help="max review image width px")
    args = ap.parse_args()

    images_dir = args.src / "images"
    labels_dir = args.src / args.label_dir
    exts = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}

    # confirmed-empty frames = negatives
    negs = []
    for lf in sorted(labels_dir.glob("*.txt")):
        if lf.stat().st_size > 0:
            continue
        img = next((images_dir / f"{lf.stem}{e}" for e in exts
                    if (images_dir / f"{lf.stem}{e}").exists()), None)
        if img:
            negs.append(img)
    print(f"confirmed-empty gommapps frames (negatives): {len(negs)}")

    out_img = args.out / "images"
    out_lbl = args.out / "labels"
    out_rev = args.out / "review"
    for d in (out_img, out_lbl, out_rev):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    records = []
    for img in negs:
        r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz,
                           device=args.device, verbose=False)[0]
        boxes = r.boxes
        n_fp = 0 if boxes is None else len(boxes)
        confs = [] if boxes is None else [float(c) for c in boxes.conf]
        max_c = max(confs) if confs else 0.0
        xyxy = [] if boxes is None else [[float(v) for v in b] for b in boxes.xyxy]
        # season from filename token TO17Su / TO18Fa / TO18Wi
        season = next((t for t in img.stem.split("_") if t.startswith("TO")), "NA")
        records.append(dict(stem=img.stem, img=img, season=season,
                            n_fp=n_fp, max_c=max_c, xyxy=xyxy))

    # hardest first: many false dets, then high confidence
    records.sort(key=lambda d: (d["n_fp"], d["max_c"]), reverse=True)

    for rank, rec in enumerate(records, 1):
        img = rec["img"]
        link = out_img / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
        (out_lbl / f"{img.stem}.txt").write_text("")
        # review render (downscaled) with false detections drawn
        with Image.open(img) as im:
            im = im.convert("RGB")
            scale = min(1.0, args.review_max / im.width)
            if scale < 1.0:
                im = im.resize((int(im.width * scale), int(im.height * scale)))
            draw = ImageDraw.Draw(im)
            for x1, y1, x2, y2 in rec["xyxy"]:
                draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                               outline=(255, 0, 0), width=3)
            im.save(out_rev / f"rank{rank:03d}_{rec['n_fp']}fp_{img.stem}.jpg", quality=85)

    with (args.out / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "stem", "season", "n_false_pos", "max_conf"])
        for rank, rec in enumerate(records, 1):
            w.writerow([rank, rec["stem"], rec["season"], rec["n_fp"], f"{rec['max_c']:.3f}"])

    n_hard = sum(1 for r in records if r["n_fp"] > 0)
    print(f"pool -> {args.out}")
    print(f"  total negatives: {len(records)}  |  model false-fires on (HARD): {n_hard}"
          f"  |  clean (no false det): {len(records) - n_hard}")
    by_season = {}
    for r in records:
        by_season[r["season"]] = by_season.get(r["season"], 0) + 1
    print("  by season:", by_season)


if __name__ == "__main__":
    main()
