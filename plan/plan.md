# ============================================================
# PROJECT PLAN (approved thesis + 6 experiments) — 2026-07-26
# ============================================================
# Thesis: aerial cetacean detection reliability governed by DATA COMPOSITION +
# SCALE-AWARE INFERENCE, not raw model capacity. One detector along a GSD axis:
# high-GSD = exact cetacean detection (mAP); low-GSD (MAMMALS) = screening/cueing
# for experts, maximize recall at low false-alarm (recall@FPPI / FROC).
# Pillars: (1) data curation/composition, (2) scale analysis + adaptive SAHI,
# (3) HITL pseudo-labeling. Proposer-confirmer ensemble PARKED.
#
# Locked decisions:
# - Positive class = cetacean-only. Seals NOT a confirmed negative (drop seal-as-neg).
# - MAMMALS = TEST-ONLY, fully held out. NO negative mining from it (leak risk).
#   Two-part test: (a) small human-boxed subset for mAP-lite (localization-tolerant),
#   (b) presence+count with injected negatives for precision/FPPI.
# - Label Studio: 2 tiers per box = certain / inferred (+ floor rule: don't box pure
#   is-it-even-an-animal specks; keep those frames out of neg pool too). No mirroring
#   of MAMMALS Definite/Probable/Possible.
# - gommapps = TRAIN + its own source-disjoint held-out TEST. Hard negatives mined
#   from gommapps NO-SIGHTING frames (CSV) at the SAHI tile size (~512-1024px TBD).
# - Hard-neg dose ~20% of positives to start; swept in E4.
# - Pre-label gommapps + MAMMALS with cetacean_yolo11n.pt (low conf) -> user corrects.
#
# 6 EXPERIMENTS (change ONE factor, fixed seed/config):
#   Baseline B = aerial-only + certain + porpoises-IN + 20% hardneg, YOLO11n.
#   E0 model size (n vs s/m) — capacity control.
#   E1 +surface vs aerial-only.
#   E2 porpoises IN vs OUT (shortcut risk).
#   E3 DROPPED 2026-08-03 (user decision). Certain/inferred tiers were not cleanly
#      separated during labeling, so a certain-only vs certain+inferred contrast is
#      not meaningful; all builds use labels_with_inferred. Out of scope.
#   E4 hardneg dose 0/20/40%.
#   E5 adaptive SAHI vs best fixed tile (inference-only, altitude benchmark).
#      DONE 2026-08-03: (A) synthetic altitude sweep runs/e5/sweep + (B) real 151MP APEM anchor
#      runs/e5/apem_{E0_s,B}. Headline: on APEM plain=0/3, adaptive=3/3 ~0FP @35 tiles (17x cheaper
#      than fixed640). Sweep: adaptive matches fixed recall at fewer tiles (mid/high alt). See
#      docs/MODEL_REGISTRY.md "E5" section. Models B + E0_s.
# Metrics: precision regime -> mAP on held-out aerial test; screening regime ->
# recall@FPPI on MAMMALS + injected gommapps negatives.
#
# STANDING RULES:
# - BEFORE STARTING EACH PHASE: stop, restate the phase, and confirm with user
#   before doing any work — allow changes first.
# - Do NOT start any TRAINING until ALL wanted labeling (gommapps + MAMMALS) is done.
#   Non-training automation (analysis, splits, neg mining, dataset prep) MAY run in
#   parallel with user labeling.
# - Pre-label confidence LOW (~0.15) so user mostly corrects, rarely draws new boxes.
# - Pre-label BOTH gommapps AND MAMMALS with cetacean_yolo11n.pt.
#
# REORDERED PHASES (labeling first, training last):
# EXECUTION LANES:
#  USER (manual, Label Studio): correct pre-labeled gommapps (certain/inferred + floor
#    rule); box MAMMALS test subset. Runs parallel with agent non-training phases.
#  AGENT (auto): everything else.
#
# Phase 1 = LABELING SETUP + PRE-LABEL (auto setup -> user).  [NEW FIRST]
#   1a. Quick box-scale calc on 9 normalized sources JUST to pick SAHI tile size
#       (fast, no training) — feeds MAMMALS SAHI pre-label + neg mining. If terminal
#       unavailable, pre-label MAMMALS at default 1024 tile, refine later.
#   1b. LS project + 2-tier config (certain/inferred) + written annotation guideline.
#   1c. Pre-label gommapps AND MAMMALS via predict_to_labelstudio.py + cetacean_yolo11n.pt
#       at conf~0.15 (SAHI for high-res MAMMALS). HANDOFF -> user labels (Lane A).
# Phase 2 = FULL SCALE ANALYSIS + SPLITS (auto; may run while user labels).
#   Split in two (gommapps NOT needed to set tile size — small count + mid-scale;
#   tile size is set by the SMALL end = porpoises/MAMMALS regime):
#   2a PRELIMINARY (before/at start, no labels): scale analysis on 9 normalized
#      sources -> pick working SAHI tile size + first figure. Feeds pre-label + neg
#      mining. (This is the same calc as Phase 1a.)
#      [DONE 2026-07-26: overall box sqrt-area median=43.6px, p5=17.5, p95=209.5
#      over 53,289 boxes (dominated by 46k uav-porpoise boxes). LOCKED working
#      tile=640, overlap=0.2, model imgsz=1024 (median obj ~6.8% of tile).]
#   2b FINAL FIGURE (after labeling): re-run with gommapps + boxed MAMMALS subset
#      folded in for the paper (honest deployment-GSD figure). Cheap re-run; tile size
#      unlikely to change. If it shifts materially -> re-mine negatives / re-pre-label.
#      [DONE 2026-07-27: scripts/box_scale_analysis.py folded gommapps in ->
#      runs/analysis/box_scale_2b.{png,csv}. Overall median 44.9px (was 43.6), p95
#      355.5px; median=7.0% of tile640 -> TILE 640 CONFIRMED. gommapps median 328.6px
#      p95 590px (fits < tile). MAMMALS boxes pending user labeling.]
#   Also: fixed source-held-out splits (aerial precision test, val, train pools);
#   porpoise near-dup subsample stride.
#   [SPLITS DONE v1 2026-07-27: scripts/make_splits.py -> data/splits/splits_v1.csv
#    (manifest only; seed 1337; ratios 80/15/5). Grouping keys from filenames:
#    gommapps season+Sight### (Misc=singleton), uav __scene__, whale-bbox-layer
#    Drone-Baleia-N. PRECISION TEST = whale-bbox-layer whole source, every5th/scene
#    (106f/153b). SCREENING TEST = MAMMALS (separate, still labeling; excluded here).
#    aerial-right-whale KEPT IN TRAIN (standalone frames). gommapps empty frames ->
#    TRAIN as hard-negatives; test biased low-but-nonzero. uav scene-split+every5th.
#    Totals kept train7803/val1417/test579. KNOWN: gommapps test small (18 boxes),
#    supplementary; dryad/whaleshape same-DOI overlap -> md5 dedup at Phase4.]
#   STEP 0 (LS export) DONE 2026-07-27: scripts/ls_export_to_normalized.py; gommapps
#    proj7 -> data/normalized/gommapps/{images,labels(certain69),labels_with_inferred
#    (all1874)}; 420 annotated (57 empty), 128 unreviewed EXCLUDED. TIER NOTE: user
#    didn't carefully split certain/inferred -> certain=clean subset, inferred=noisy
#    (many true certains); labels_with_inferred = realistic set used for splits.
#   PORPOISE SUBSAMPLING (user: every 5th/10th frame per scene):
#   - Does NOT affect 2a tile size (porpoise box SIZE is stride-invariant; only count
#     changes). Stride is a Phase 2-split / Phase 4-build decision.
#   - LEAKAGE RULE: uav-porpoises split must be SCENE-LEVEL (whole scene -> one of
#     train/val/test), THEN subsample within each side. Never frame-level random split
#     (consecutive frames near-identical -> leak).
#   - 2b final scale figure computed on the SUBSAMPLED training set (honest weighting).
#     Per-box size-distribution shape is stride-invariant; only porpoise weight shrinks.
#   - Stride interacts with E2 (porpoise dominance / bright-speck shortcut risk).
# Phase 3 = HARD-NEG MINING (auto; parallel w/ user). gommapps no-sighting frames ->
#   negative tiles at tile size. NO MAMMALS mining (leak).
#   [DONE 2026-07-27: DATA REALITY = no large no-sighting pool; raw gommapps=548 imgs
#    all in LS; CSVs don't map to extra imagery. Only confirmed negs = 57 user-empty
#    frames. Negatives kept FULL-SIZE (user: close-up imagery, not tiled) to match
#    gommapps positives. scripts/mine_hard_negatives.py (pre-label model conf0.10) ->
#    data/hard_negatives/gommapps_fullframe/{images,labels(empty),review,manifest.csv}.
#    57 negs: 8 HARD (model false-fires), 49 clean; ranked hardest-first for review.
#    Old data/hard_negatives/{gommapps,gommapps_raw,apem} = stale TILED, superseded.
#    NOTE: MAMMALS is test-only -> Ph4 build + Ph5 train DON'T need it (only Ph6 eval).]
#   [TILES 2026-07-28: user reuses prior gommapps tiles. Mixing full-size+tiled negs OK.
#    scripts/validate_negative_tiles.py: keep tile only if parent in gommapps TRAIN split
#    AND tile rect (tile img WxH + filename _x_y) doesn't overlap any GT box. 168 cand ->
#    KEPT 80 (train-safe, whale-free); dropped 70 parent_not_train + 18 overlaps_whale ->
#    data/hard_negatives/gommapps_tiles_clean/. COMBINED NEG POOL = 57 fullframe+80 tiles
#    = 137.]
# Phase 4 = DATASET BUILD (auto; AFTER labeling complete). build_dataset.py ->
#   E0-E5 composition data.yaml variants; fold in gommapps labels + MAMMALS test boxes.
# Phase 5 = TRAINING (auto; ONLY after ALL labeling done). Baseline B + E0-E5, fixed seed.
# Phase 6 = EVALUATION (auto). mAP (aerial test) + recall@FPPI/FROC (MAMMALS+injected
#   gommapps negatives) + adaptive-SAHI altitude benchmark (E5); tables+figures.
# Phase 7 = PAPER (auto + user review). cetacean-aerial-detection/paper via tectonic.
#
# CONSIDERATIONS RESOLVED:
# - NO early training (user override of prior suggestion). Training waits for all labels.
# - Pre-label conf lowered to ~0.15.
# - MAMMALS also pre-labeled with existing model.
#
# Reusable assets: cetacean-aerial-detection/models/cetacean_yolo11n.pt (pre-label),
#   scripts/predict_to_labelstudio.py, frames_to_labelstudio.py; build_dataset.py from
#   /home/dell/cetacean_detection/scripts (adapt). draw_gt.py already copied.
# ============================================================

# Plan: Normalize all labeled raw sources (cetacean-detection-final)

Goal now = get every LABELED source into uniform single-class YOLO under
data/normalized/ + manifest.csv. Normalizing is STAGING only — it does not commit a
source to training; inclusion + splits are decided later at build time. Then STOP and
discuss dataset build/splits separately.

## Scope (user: "everything labeled" + visual spot-check)
Trustworthy now (9): aerial-right-whale, whale-bbox-layer, dryad, uav-porpoises,
cetacean-detection, cetacean-detector, dolphin, dolphin2 (8 script-ready) + WHALESHAPE
(dryad_6q573n668, needs COCO converter).
Review-gated (2): whale-321z9 (nc=3 names '0/1/2'), whale-jybhw (nc=1 'object-detection').
Generic classes -> can't trust a blind collapse-to-cetacean; must eyeball GT first.

## Phase 1 — code additions to scripts/normalize_labels.py
1. Add `convert_coco` (WHALESHAPE). Schema = JSON-lines, one rec/line:
   {image_id, file_name, annotations:[{bbox:[x,y,w,h], bbox_mode:1 (XYWH abs px),
   category_id:0, segmentation, keypoints}]}. Images in images/ subdir. Read W/H via
   PIL, convert xywh-abs -> yolo norm (drop seg/keypoints). Split: train.json->train,
   test.json->test.
2. Add SOURCES entry "whaleshape" (domain aerial, format coco,
   path aerial/labeled/dryad_doi_10_5061_dryad_6q573n668__v20260312).
3. Add optional per-source `keep_classes` filter to convert_roboflow (default: keep all)
   — needed only if 321z9/jybhw review shows non-cetacean classes must be dropped.
4. Register "coco" in CONVERTERS.

## Phase 2 — run + verify trustworthy 9
5. Run: ml-venv/bin/python scripts/normalize_labels.py (all trustworthy sources).
6. Copy scripts/draw_gt.py from cetacean_detection; render GT contact sheets per source
   -> runs/viz/gt_review/<source>/. Confirm: dryad EXIF-corrected (boxes on whales, not
   90/180 off; expect exif_reoriented>0), WHALESHAPE boxes land on whales (COCO ok),
   uav tiny porpoise boxes correct.
7. Check manifest.csv: per-source images/boxes, dedup_skipped, aug_collapsed totals.

## Phase 3 — review-gated 321z9 + jybhw  [DONE -> BOTH DROPPED]
Reviewed via class-colored montages. Finding: ~1 box/image, and each box marks a
single anatomical PART (321z9: 0=body/1=dorsal fin[dominant 4246]/2=fluke;
jybhw: single class = body-or-fin). Not whole-animal boxes -> unsuitable for a
whole-animal detector. User decision: DROP both. Neither is in SOURCES; docstring
updated to record the drop. merge_part_boxes.py removed (merge approach abandoned:
0 merges since boxes don't co-occur per image).

## Then PAUSE -> discuss dataset build + splits (Phase 4, not detailed yet)
Open questions for that discussion: which normalized sources go in; surface-in-train
vs aerial-only; val/test = aerial-only? uav subsample stride; WHALESHAPE vs dryad
leakage (md5 dedup + fixed test); backgrounds/hard-negatives; fixed held-out test.

## Relevant files
- scripts/normalize_labels.py (copied, RAW_ROOT already -> data/raw). Add coco converter
  + whaleshape/321z9/jybhw SOURCES + optional keep_classes. Reuse convert_via EXIF logic
  (dryad), roboflow_base aug-collapse, md5 dedup (emit()).
- scripts/draw_gt.py (COPY from /home/dell/cetacean_detection/scripts/draw_gt.py).
- data/normalized/ (output) ; data/normalized/manifest.csv.
- docs/raw_datasets.md (source inventory reference).

## Verification
- data/normalized/<source>/{images(symlinks),labels(.txt)} present for each source.
- manifest.csv rows == sum of per-source images; dedup/aug stats printed.
- GT contact sheets visually correct per source (esp dryad EXIF, WHALESHAPE COCO,
  321z9/jybhw class check).

## Decisions
- Normalize broadly (staging); prune at build. 321z9/jybhw behind a GT review gate.
- WHALESHAPE gets a COCO converter now (bbox only; seg/keypoints ignored).
