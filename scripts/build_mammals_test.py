#!/usr/bin/env python3
"""Build a YOLO test set from human-reviewed MAMMALS annotations in the Label Studio DB.

Non-cancelled completions in project 'MAMMALS' become the ground truth:
  - boxes -> class 0 (cetacean), LS percent xywh (top-left) -> YOLO normalized cxcywh
  - empty completions -> empty label file (true negative)
Images are symlinked from the LS local-files root.
"""
import json
import os
import sqlite3
import sys

LS_DB = "/home/dell/.local/share/label-studio/label_studio.sqlite3"
IMG_ROOT = "/mnt/fax/hidef/MAMMALS"
OUT = "/home/dell/cetacean-detection-final/data/dataset/test_mammals"
PID = 8  # MAMMALS

img_dir = os.path.join(OUT, "images")
lbl_dir = os.path.join(OUT, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

con = sqlite3.connect(LS_DB)
cur = con.cursor()
rows = cur.execute(
    """select tc.result, t.data
       from task_completion tc join task t on tc.task_id=t.id
       where t.project_id=? and tc.was_cancelled=0""",
    (PID,),
).fetchall()

n_img = n_box = n_empty = n_missing = 0
for result_json, data_json in rows:
    result = json.loads(result_json)
    data = json.loads(data_json)
    src = data.get("image", "")
    fname = src.split("?d=MAMMALS/")[-1] if "?d=MAMMALS/" in src else os.path.basename(src)
    src_path = os.path.join(IMG_ROOT, fname)
    if not os.path.exists(src_path):
        n_missing += 1
        print("MISSING:", src_path, file=sys.stderr)
        continue
    stem = os.path.splitext(os.path.basename(fname))[0]
    dst_img = os.path.join(img_dir, os.path.basename(fname))
    if not os.path.lexists(dst_img):
        os.symlink(src_path, dst_img)
    lines = []
    for r in result:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        x, y, w, h = v["x"], v["y"], v["width"], v["height"]
        cx = (x + w / 2) / 100.0
        cy = (y + h / 2) / 100.0
        wn = w / 100.0
        hn = h / 100.0
        # clamp
        cx, cy, wn, hn = (min(max(a, 0.0), 1.0) for a in (cx, cy, wn, hn))
        if wn <= 0 or hn <= 0:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
        n_box += 1
    with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
        f.write("\n".join(lines))
    if not lines:
        n_empty += 1
    n_img += 1

print(f"images={n_img} boxes={n_box} empty={n_empty} missing={n_missing}")
