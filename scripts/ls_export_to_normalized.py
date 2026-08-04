#!/usr/bin/env python3
"""Export a Label Studio project's annotations -> normalized single-class YOLO.

Pulls the project via the REST API, takes each task's latest annotation, and writes
YOLO label files (class 0 = cetacean) plus image symlinks under an output source dir.

The 2-tier label (certain/inferred) is preserved as two parallel label dirs:
  labels/                -> certain boxes only        (baseline B / certain-only)
  labels_with_inferred/  -> certain + inferred boxes  (E3 certain+inferred variant)

A submitted-but-empty annotation (a frame the human cleared of boxes) becomes an
empty label file in both dirs = a valid negative frame. Tasks with NO annotation
are skipped and reported (not yet reviewed).
"""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import requests


def parse_local_path(image_url, doc_root):
    """/data/local-files/?d=<rel> -> absolute path under doc_root."""
    q = parse_qs(urlparse(image_url).query)
    rel = unquote(q.get("d", [""])[0])
    return Path(doc_root) / rel


def to_yolo(item):
    """LS rectanglelabels (percent top-left) -> (cx,cy,w,h) normalized center, + tier."""
    v = item["value"]
    x, y, w, h = v["x"], v["y"], v["width"], v["height"]
    cx = (x + w / 2) / 100.0
    cy = (y + h / 2) / 100.0
    tier = (v.get("rectanglelabels") or ["certain"])[0]
    return cx, cy, w / 100.0, h / 100.0, tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--token", required=True)
    ap.add_argument("--project", type=int, required=True)
    ap.add_argument("--doc-root", required=True,
                    help="LOCAL_FILES_DOCUMENT_ROOT used by the LS server.")
    ap.add_argument("--out", required=True, help="Output normalized source dir.")
    args = ap.parse_args()

    h = {"Authorization": f"Token {args.token}"}
    r = requests.get(f"{args.base}/api/projects/{args.project}/export",
                     headers=h, params={"exportType": "JSON"})
    r.raise_for_status()
    tasks = r.json()

    out = Path(args.out)
    img_dir = out / "images"
    lbl_cert = out / "labels"
    lbl_all = out / "labels_with_inferred"
    for d in (img_dir, lbl_cert, lbl_all):
        d.mkdir(parents=True, exist_ok=True)

    n_ann = n_empty = n_skip = 0
    n_cert = n_inf = 0
    for t in tasks:
        anns = t.get("annotations") or []
        anns = [a for a in anns if not a.get("was_cancelled")]
        if not anns:
            n_skip += 1
            continue
        result = anns[-1].get("result", [])
        img_path = parse_local_path(t["data"]["image"], args.doc_root)
        if not img_path.exists():
            print(f"  WARN missing image: {img_path}")
            continue
        stem = img_path.stem
        # symlink image
        link = img_dir / img_path.name
        if not link.exists():
            os.symlink(img_path.resolve(), link)
        cert_lines, all_lines = [], []
        for it in result:
            if it.get("type") != "rectanglelabels":
                continue
            cx, cy, w, hh, tier = to_yolo(it)
            line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {hh:.6f}"
            all_lines.append(line)
            if tier == "certain":
                cert_lines.append(line)
                n_cert += 1
            else:
                n_inf += 1
        (lbl_cert / f"{stem}.txt").write_text("\n".join(cert_lines))
        (lbl_all / f"{stem}.txt").write_text("\n".join(all_lines))
        n_ann += 1
        if not all_lines:
            n_empty += 1

    print(f"project {args.project} -> {args.out}")
    print(f"  annotated frames written: {n_ann} (empty/negative: {n_empty})")
    print(f"  boxes: certain={n_cert}  inferred={n_inf}  total={n_cert + n_inf}")
    print(f"  tasks skipped (no annotation): {n_skip}")


if __name__ == "__main__":
    main()
